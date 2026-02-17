#!/usr/bin/env python3
"""
lit_review.py — ArXiv literature review: search, rank, download.

Usage:
    python3 lit_review.py search "query" --dir /path [--max-papers 50] [--abstracts-only]
    python3 lit_review.py cleanup --dir /path --keep 50 "overall topic description"
    python3 lit_review.py list --dir /path
"""

import json
import re
import shutil
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_DELAY = 3  # seconds between arXiv requests


# ── Slugify ────────────────────────────────────────────────────────────

def slugify(text, max_len=50):
    """Convert search terms to a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r'[^a-z0-9\s-]', '', s)
    s = re.sub(r'[\s]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s[:max_len]


# ── ArXiv API ──────────────────────────────────────────────────────────

def search_arxiv(query_terms, max_results=200):
    """Search arXiv API and return list of paper metadata dicts."""
    candidates = []
    page_size = 100
    pages = (max_results + page_size - 1) // page_size

    for page in range(pages):
        start = page * page_size
        remaining = max_results - start
        fetch = min(page_size, remaining)

        params = urllib.request.quote(query_terms)
        url = f"{ARXIV_API}?search_query=all:{params}&start={start}&max_results={fetch}&sortBy=relevance"

        if page > 0:
            print(f"  Waiting {ARXIV_DELAY}s before next API page...")
            time.sleep(ARXIV_DELAY)

        print(f"  Fetching arXiv results {start+1}-{start+fetch}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'lit-review-tool/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read().decode('utf-8')
        except Exception as e:
            print(f"  Warning: arXiv API request failed: {e}")
            break

        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entries = root.findall('atom:entry', ns)
        if not entries:
            break

        for entry in entries:
            title = entry.find('atom:title', ns)
            abstract = entry.find('atom:summary', ns)
            published = entry.find('atom:published', ns)
            arxiv_id_el = entry.find('atom:id', ns)

            if title is None or abstract is None or arxiv_id_el is None:
                continue

            # Extract arXiv ID from URL like http://arxiv.org/abs/2401.12345v1
            raw_id = arxiv_id_el.text.strip()
            arxiv_id = raw_id.split('/abs/')[-1]
            # Strip version suffix for PDF URL
            base_id = re.sub(r'v\d+$', '', arxiv_id)

            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text.strip())

            # Find PDF link
            pdf_url = f"https://arxiv.org/pdf/{base_id}"
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                    break

            candidates.append({
                'arxiv_id': arxiv_id,
                'base_id': base_id,
                'title': ' '.join(title.text.strip().split()),
                'abstract': ' '.join(abstract.text.strip().split()),
                'authors': authors,
                'published': published.text.strip() if published is not None else '',
                'pdf_url': pdf_url,
            })

        if len(entries) < fetch:
            break  # No more results

    return candidates


# ── Embedding-based ranking ───────────────────────────────────────────

def rank_by_relevance(query_text, candidates, openai_client, top_n=50):
    """Rank candidates by cosine similarity to query embedding."""
    if not candidates:
        return []

    texts = [query_text] + [c['abstract'] for c in candidates]
    # Truncate to stay within token limits
    texts = [t[:8000] for t in texts]

    print(f"  Embedding {len(texts)} texts for relevance ranking...")
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])

    query_emb = all_embeddings[0]
    abstract_embs = all_embeddings[1:]

    # Cosine similarity (embeddings are already normalized by OpenAI)
    scored = []
    for i, emb in enumerate(abstract_embs):
        dot = sum(a * b for a, b in zip(query_emb, emb))
        scored.append((dot, i))

    scored.sort(reverse=True)
    selected = []
    for sim, idx in scored[:top_n]:
        paper = dict(candidates[idx])
        paper['similarity'] = round(sim, 4)
        selected.append(paper)

    return selected


# ── Search command ─────────────────────────────────────────────────────

def cmd_search(query_terms, output_dir, max_papers=50, abstracts_only=False):
    """Full search workflow: arXiv → rank → download."""
    openai_client = OpenAI()

    # Set up output directory
    review_dir = Path(output_dir)
    papers_dir = review_dir / "papers"

    review_dir.mkdir(parents=True, exist_ok=True)
    papers_dir.mkdir(exist_ok=True)

    print(f"=== Literature Review: {query_terms} ===")
    print(f"Directory: {review_dir}\n")

    print("[Phase 1/3] Searching arXiv...")
    candidates = search_arxiv(query_terms, max_results=200)
    print(f"  Found {len(candidates)} candidates\n")

    if not candidates:
        print("No results found on arXiv. Try different search terms.")
        shutil.rmtree(review_dir)
        return

    candidates_path = review_dir / "candidates.json"
    with open(candidates_path, 'w') as f:
        json.dump(candidates, f, indent=2)

    print("[Phase 2/3] Ranking by abstract relevance...")
    selected = rank_by_relevance(query_terms, candidates, openai_client, top_n=max_papers)
    print(f"  Selected top {len(selected)} papers")
    if selected:
        print(f"  Similarity range: {selected[-1]['similarity']:.3f} - {selected[0]['similarity']:.3f}")
    print()

    # Merge with existing selected_papers.json (accumulate across runs)
    selected_path = review_dir / "selected_papers.json"
    existing = {}
    if selected_path.exists():
        with open(selected_path) as f:
            for p in json.load(f):
                existing[p.get('base_id', p.get('arxiv_id'))] = p

    for p in selected:
        key = p.get('base_id', p.get('arxiv_id'))
        if key not in existing or p.get('similarity', 0) > existing[key].get('similarity', 0):
            existing[key] = p

    merged = list(existing.values())
    merged.sort(key=lambda p: p.get('similarity', 0), reverse=True)
    with open(selected_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"  Accumulated {len(merged)} unique papers in selected_papers.json")

    # Print top papers
    print("  Top 10 papers:")
    for i, p in enumerate(selected[:10]):
        print(f"    {i+1}. [{p['similarity']:.3f}] {p['title'][:80]}")
    print()

    # Download PDFs (unless abstracts-only)
    downloaded = 0
    if abstracts_only:
        print("[Phase 3/3] Skipping PDF download (--abstracts-only)")
    else:
        print(f"[Phase 3/3] Downloading {len(selected)} PDFs...")
        for i, paper in enumerate(selected):
            pdf_path = papers_dir / f"{paper['base_id'].replace('/', '_')}.pdf"
            if pdf_path.exists():
                downloaded += 1
                continue

            print(f"  [{i+1}/{len(selected)}] Downloading {paper['base_id']}...")
            if i > 0:
                time.sleep(ARXIV_DELAY)

            try:
                req = urllib.request.Request(
                    paper['pdf_url'],
                    headers={'User-Agent': 'lit-review-tool/1.0'}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(pdf_path, 'wb') as f:
                        f.write(resp.read())
                downloaded += 1
            except Exception as e:
                print(f"    Warning: download failed: {e}")

        print(f"  Downloaded {downloaded}/{len(selected)} PDFs\n")

    # Summary
    total_size = sum(
        f.stat().st_size for f in review_dir.rglob('*') if f.is_file()
    )
    size_mb = total_size / (1024 * 1024)

    today = date.today().isoformat()
    manifest = {
        'query': query_terms,
        'created': today,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'candidates_count': len(candidates),
        'selected_count': len(selected),
        'downloaded_count': downloaded,
        'abstracts_only': abstracts_only,
        'disk_usage_mb': round(size_mb, 1),
        'directory': str(review_dir),
    }
    with open(review_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Literature review complete!")
    print(f"  Query:      {query_terms}")
    print(f"  Papers:     {len(selected)} selected from {len(candidates)} candidates")
    print(f"  Downloaded: {downloaded}")
    print(f"  Disk usage: {size_mb:.1f} MB")
    print(f"  Directory:  {review_dir}")
    print(f"\nTo build a knowledge graph:")
    print(f"  /knowledge-graph {review_dir}")
    print(f"{'='*60}")


# ── Cleanup command ────────────────────────────────────────────────────

def cmd_cleanup(review_dir_str, query_terms, keep=50):
    """Re-rank all papers against a unified query, keep top N, delete the rest."""
    review_dir = Path(review_dir_str)
    papers_dir = review_dir / "papers"
    selected_path = review_dir / "selected_papers.json"

    if not selected_path.exists():
        print(f"Error: no selected_papers.json in {review_dir}")
        sys.exit(1)

    with open(selected_path) as f:
        all_papers = json.load(f)

    if not all_papers:
        print("No papers to clean up.")
        return

    print(f"=== Cleanup: re-ranking {len(all_papers)} papers ===")
    print(f"  Query: {query_terms}")
    print(f"  Keep:  {keep}\n")

    # Re-rank all papers against the unified query
    openai_client = OpenAI()
    ranked = rank_by_relevance(query_terms, all_papers, openai_client, top_n=len(all_papers))

    # Split into keep vs remove
    kept = ranked[:keep]

    # Delete PDFs not in the kept set (single pass)
    kept_stems = {p.get('base_id', '').replace('/', '_') for p in kept}
    deleted_count = 0
    deleted_bytes = 0
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        if pdf_path.stem not in kept_stems:
            size = pdf_path.stat().st_size
            pdf_path.unlink()
            deleted_count += 1
            deleted_bytes += size

    # Write updated selected_papers.json with only kept papers
    with open(selected_path, 'w') as f:
        json.dump(kept, f, indent=2)

    # Update manifest
    total_size = sum(
        f.stat().st_size for f in review_dir.rglob('*') if f.is_file()
    )
    size_mb = total_size / (1024 * 1024)
    saved_mb = deleted_bytes / (1024 * 1024)

    manifest_path = review_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest['selected_count'] = len(kept)
    manifest['cleanup_query'] = query_terms
    manifest['cleanup_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest['disk_usage_mb'] = round(size_mb, 1)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Cleanup complete!")
    print(f"  Kept:    {len(kept)} papers")
    print(f"  Removed: {deleted_count} PDFs ({saved_mb:.1f} MB freed)")
    print(f"  Disk:    {size_mb:.1f} MB remaining")
    if kept:
        print(f"  Similarity range: {kept[-1].get('similarity', 0):.3f} - {kept[0].get('similarity', 0):.3f}")
    print(f"\nTo build a knowledge graph:")
    print(f"  /knowledge-graph {review_dir}")
    print(f"{'='*60}")


# ── List command ───────────────────────────────────────────────────────

def cmd_list(scan_dir):
    """List all existing literature reviews in a directory."""
    scan_path = Path(scan_dir)
    reviews = sorted(scan_path.glob("*/manifest.json"))

    if not reviews:
        print(f"No literature reviews found in {scan_path}")
        return

    print(f"{'Name':<45} {'Papers':>6} {'Size':>8}  Query")
    print("-" * 80)

    for manifest_path in reviews:
        with open(manifest_path) as f:
            m = json.load(f)
        name = manifest_path.parent.name
        mode = " [abs]" if m.get('abstracts_only') else ""
        print(f"{name:<45} {m.get('selected_count', '?'):>6} {m.get('disk_usage_mb', 0):>6.1f}MB  {m.get('query', '?')}{mode}")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or args[0] in ('--help', '-h'):
        print(__doc__)
        return

    cmd = args[0]

    if cmd == 'search':
        # Parse: search "query" --dir <path> [--max-papers N] [--abstracts-only]
        query_parts = []
        max_papers = 50
        abstracts_only = False
        output_dir = None
        i = 1
        while i < len(args):
            if args[i] == '--max-papers' and i + 1 < len(args):
                max_papers = int(args[i + 1])
                i += 2
            elif args[i] == '--dir' and i + 1 < len(args):
                output_dir = args[i + 1]
                i += 2
            elif args[i] == '--abstracts-only':
                abstracts_only = True
                i += 1
            else:
                query_parts.append(args[i])
                i += 1
        query_text = ' '.join(query_parts)
        if not query_text:
            print("Error: provide search terms after 'search'")
            sys.exit(1)
        if not output_dir:
            print("Error: --dir <path> is required")
            sys.exit(1)
        cmd_search(query_text, output_dir=output_dir, max_papers=max_papers, abstracts_only=abstracts_only)

    elif cmd == 'cleanup':
        # Parse: cleanup --dir <path> --keep N "query terms"
        output_dir = None
        keep = 50
        query_parts = []
        i = 1
        while i < len(args):
            if args[i] == '--dir' and i + 1 < len(args):
                output_dir = args[i + 1]
                i += 2
            elif args[i] == '--keep' and i + 1 < len(args):
                keep = int(args[i + 1])
                i += 2
            else:
                query_parts.append(args[i])
                i += 1
        if not output_dir:
            print("Error: --dir <path> is required")
            sys.exit(1)
        query_text = ' '.join(query_parts)
        if not query_text:
            print("Error: provide a query describing the overall topic")
            sys.exit(1)
        cmd_cleanup(output_dir, query_text, keep=keep)

    elif cmd == 'list':
        # Parse: list --dir <path>
        scan_dir = None
        i = 1
        while i < len(args):
            if args[i] == '--dir' and i + 1 < len(args):
                scan_dir = args[i + 1]
                i += 2
            else:
                i += 1
        if not scan_dir:
            print("Error: --dir <path> is required (directory containing review subdirectories)")
            sys.exit(1)
        cmd_list(scan_dir)

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: search, cleanup, list")
        sys.exit(1)


if __name__ == '__main__':
    main()
