#!/usr/bin/env python3
"""
build_knowledge_graph.py — Ingest PDFs and build a knowledge graph.

Given a directory with papers/*.pdf (e.g. from /lit-review), this script:
1. Ingests PDFs into ChromaDB (if not already done)
2. Reads chunks from ChromaDB
3. Extracts concepts/relationships via GPT-4o-mini
4. Generates an interactive D3.js HTML visualization

Usage:
    python3 build_knowledge_graph.py --dir <path>           # Full pipeline
    python3 build_knowledge_graph.py --dir <path> --resume  # Resume interrupted build
    python3 build_knowledge_graph.py --dir <path> --viz-only # Regenerate HTML only
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from kg.config import COLLECTION_NAMES, INGEST_COLLECTION
from kg.extract import extract_concepts, select_representative_chunks
from kg.graph import build_graph, merge_extractions
from kg.ingest import extract_text_from_pdf, chunk_text, get_embeddings, ingest_files
from kg.visualize import generate_html


# ── Read chunks from ChromaDB ─────────────────────────────────────────

def load_chunks_from_rag(rag_dir):
    """Load all chunks from a ChromaDB index, grouped by paper."""
    import chromadb

    chroma_dir = rag_dir / "chroma_db"
    if not chroma_dir.exists():
        print(f"Error: no ChromaDB found at {chroma_dir}")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(chroma_dir))

    collection = None
    for name in COLLECTION_NAMES:
        try:
            collection = client.get_collection(name)
            break
        except Exception:
            continue

    if collection is None:
        cols = client.list_collections()
        if cols:
            collection = cols[0]
        else:
            print(f"Error: no collections found in {chroma_dir}")
            sys.exit(1)

    total = collection.count()
    print(f"  ChromaDB collection '{collection.name}': {total} chunks")

    papers = defaultdict(list)
    batch_size = 1000
    for offset in range(0, total, batch_size):
        limit = min(batch_size, total - offset)
        results = collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(results["documents"], results["metadatas"]):
            source = meta.get("source", "unknown")
            chunk_index = meta.get("chunk_index", 0)
            papers[source].append({
                "text": doc,
                "chunk_index": chunk_index,
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
            })

    for source in papers:
        papers[source].sort(key=lambda c: c["chunk_index"])

    print(f"  {len(papers)} papers found")
    return papers


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    rag_dir = None
    resume = False
    viz_only = False

    i = 0
    while i < len(args):
        if args[i] == '--dir' and i + 1 < len(args):
            rag_dir = Path(args[i + 1])
            i += 2
        elif args[i] == '--resume':
            resume = True
            i += 1
        elif args[i] == '--viz-only':
            viz_only = True
            i += 1
        else:
            i += 1

    if rag_dir is None:
        print("Error: --dir <path> is required")
        print(__doc__)
        sys.exit(1)

    graph_file = rag_dir / "knowledge_graph.json"
    html_file = rag_dir / "knowledge_graph.html"
    cache_file = rag_dir / "extraction_cache.json"

    # Viz-only mode: regenerate HTML from existing graph
    if viz_only:
        if not graph_file.exists():
            print(f"Error: no knowledge_graph.json in {rag_dir}")
            sys.exit(1)
        with open(graph_file) as f:
            graph = json.load(f)
        title = f"Knowledge Graph ({graph['metadata']['total_concepts']}c, {graph['metadata']['total_edges']}e)"
        html, n_nodes, n_links = generate_html(graph, title)
        with open(html_file, 'w') as f:
            f.write(html)
        print(f"Visualization: {n_nodes} nodes, {n_links} edges -> {html_file}")
        import webbrowser
        webbrowser.open(f"file://{html_file}")
        return

    # Auto-detect: if chroma_db/ missing but papers/ has PDFs, ingest first
    chroma_dir = rag_dir / "chroma_db"
    papers_dir = rag_dir / "papers"
    if not chroma_dir.exists() and papers_dir.exists() and list(papers_dir.glob("*.pdf")):
        print(f"No ChromaDB found — ingesting PDFs from {papers_dir}...")
        openai_client = OpenAI()
        pdfs = sorted(papers_dir.glob("*.pdf"))

        # Load paper metadata if available
        metadata_map = {}
        selected_path = rag_dir / "selected_papers.json"
        if selected_path.exists():
            with open(selected_path) as f:
                for paper in json.load(f):
                    key = paper.get('base_id', '').replace('/', '_') + '.pdf'
                    metadata_map[key] = paper

        ingest_files(pdfs, chroma_dir, openai_client, metadata_map)
        print()

    # Load chunks from RAG
    print(f"Reading RAG at {rag_dir}...")
    papers = load_chunks_from_rag(rag_dir)

    # Load cache if resuming
    cache = {}
    if resume and cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Resuming: {len(cache)} papers already processed.\n")

    # Extract from each paper
    client = OpenAI()
    for paper_name, chunks in sorted(papers.items()):
        if paper_name in cache:
            continue

        selected = select_representative_chunks(chunks)
        text = "\n\n---\n\n".join(c["text"][:3000] for c in selected)
        if len(text) > 6000:
            text = text[:6000] + "\n[...truncated...]"

        extraction = extract_concepts(text, paper_name, client)
        n_c = len(extraction.get("concepts", []))
        n_r = len(extraction.get("relationships", []))
        print(f"  {paper_name[:55]:55s} -> {n_c}c, {n_r}r")

        cache[paper_name] = extraction

        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)

        time.sleep(0.5)

    # Merge
    print(f"\nMerging {len(cache)} paper extractions...")
    concepts, edges = merge_extractions(cache)
    print(f"  {len(concepts)} unique concepts, {len(edges)} relationships")

    # Build and save graph
    graph = build_graph(concepts, edges)
    with open(graph_file, 'w') as f:
        json.dump(graph, f, indent=2)

    # Generate HTML
    title = f"Knowledge Graph ({graph['metadata']['total_concepts']}c, {graph['metadata']['total_edges']}e)"
    html, n_nodes, n_links = generate_html(graph, title)
    with open(html_file, 'w') as f:
        f.write(html)

    print(f"\nKnowledge graph saved to {graph_file}")
    print(f"  {graph['metadata']['total_concepts']} concepts")
    print(f"  {graph['metadata']['total_edges']} edges")
    print(f"  {graph['metadata']['total_papers']} papers")
    print(f"\nVisualization: {n_nodes} nodes, {n_links} edges -> {html_file}")

    # Show most connected concepts
    print(f"\n{'='*60}")
    print("MOST REFERENCED CONCEPTS (by paper count)")
    print(f"{'='*60}")
    by_papers = sorted(concepts.values(),
                       key=lambda c: len(c["papers"]), reverse=True)
    for c in by_papers[:20]:
        print(f"  {c['name']:45s}  ({len(c['papers'])} papers)")

    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{html_file}")


if __name__ == "__main__":
    main()
