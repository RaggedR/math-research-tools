"""Level 1 summary generation for the INSTINCT architecture.

For each concept in the knowledge graph (degree >= min_degree), queries ChromaDB
for relevant passages and uses an LLM to write a 2-3 page structured summary.
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from .config import load_config, EMBEDDING_MODEL
from .llm import with_retry


# ── Defaults ──────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"
TOP_K = 10
MIN_DEGREE = 2
DEFAULT_COLLECTION = "lit_review"


# ── Utilities ──────────────────────────────────────────────────────────

def slugify(name):
    """Convert concept name to a filesystem-safe slug."""
    s = name.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s]+', '-', s)
    s = re.sub(r'-+', '-', s)
    return s.strip('-')[:80]


def compute_degrees(graph):
    """Compute degree for each concept node."""
    degree = defaultdict(int)
    for e in graph["edges"]:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    return degree


def filter_concepts(graph, min_degree=MIN_DEGREE):
    """Return concepts with degree >= min_degree."""
    degree = compute_degrees(graph)
    return [c for c in graph["concepts"] if degree.get(c["name"], 0) >= min_degree]


# ── RAG Query ──────────────────────────────────────────────────────────

def query_chroma(collection, openai_client, query_text, top_k=TOP_K):
    """Query ChromaDB and return relevant passages."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query_text[:8000]],
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def format_passages(results):
    """Format retrieved passages for the LLM prompt."""
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results['distances'][0]

    passages = []
    for doc, meta, dist in zip(docs, metas, dists):
        similarity = 1 - dist
        source = meta.get('source', 'unknown')
        pages = json.loads(meta.get('pages', '[]'))
        page_str = f"pp. {pages[0]}-{pages[-1]}" if pages else ""
        passages.append(
            f"[Source: {source} {page_str}, similarity: {similarity:.3f}]\n{doc.strip()}"
        )

    return "\n\n---\n\n".join(passages)


# ── Knowledge Graph Neighbors ─────────────────────────────────────────

def get_neighbors(graph, concept_name):
    """Get neighboring concepts from the knowledge graph."""
    neighbors = []
    for e in graph["edges"]:
        if e["source"] == concept_name:
            neighbors.append(f"-> [{e['relation']}] {e['target']}")
        elif e["target"] == concept_name:
            neighbors.append(f"<- [{e['relation']}] {e['source']}")
    return neighbors


# ── Summary Generation ────────────────────────────────────────────────

def build_user_prompt(concept, passages_text, neighbors, summary_sections=None):
    """Build the prompt for summary generation."""
    display_name = concept.get('display_name', concept['name'])
    neighbor_text = "\n".join(neighbors) if neighbors else "(none)"

    if summary_sections:
        sections_text = "\n\n".join(summary_sections)
    else:
        sections_text = """## Definition
Precise definition(s) of this concept. Include formal notation where appropriate.

## Methods and Approaches
Key algorithms, architectures, or techniques. Describe how they work at a high level.

## Evaluation
How this concept is typically evaluated — benchmarks, datasets, metrics, baselines.

## Applications
Real-world uses and practical implications.

## Key Results
State important findings with attribution (author, year). Include quantitative results
where available.

## Connections
How this concept relates to other concepts in the research landscape.

## References
List the source papers cited in the passages."""

    return f"""Write a reference summary for: **{display_name}**

Type: {concept.get('type', 'unknown')}
Description: {concept.get('description', 'N/A')}
Referenced in: {len(concept.get('papers', []))} papers

Related concepts in the knowledge graph:
{neighbor_text}

INSTRUCTIONS:
Write a 2-3 page summary. Include these sections:

{sections_text}

IMPORTANT:
- If the passages don't contain enough material, write what you have — don't pad.
- Use precise technical language.
- Every claim should be attributed to a specific paper where possible.
- Use LaTeX notation ($...$) for mathematical expressions only when necessary.

RETRIEVED PASSAGES FROM RESEARCH PAPERS:
{passages_text}"""


def generate_summary(openai_client, concept, passages_text, neighbors, model,
                     system_prompt="", summary_sections=None):
    """Use LLM to generate a structured summary."""
    prompt = build_user_prompt(concept, passages_text, neighbors, summary_sections)

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4000,
        temperature=0.2,
    )

    return response.choices[0].message.content


# ── Main pipeline ─────────────────────────────────────────────────────

def run_summaries(base_dir, config=None, start_idx=0, end_idx=None,
                  model=DEFAULT_MODEL, collection_name=None):
    """Run the Level 1 summary generation pipeline.

    Args:
        base_dir: Path to data directory containing knowledge_graph.json and chroma_db/
        config: Optional DomainConfig. If None, loaded from base_dir.
        start_idx: Start index for batch processing.
        end_idx: End index for batch processing (None = all).
        model: LLM model name.
        collection_name: Override ChromaDB collection name.
    """
    import chromadb
    from openai import OpenAI
    import random
    import sys

    base_dir = Path(base_dir)

    if config is None:
        config = load_config(data_dir=base_dir)

    # Resolve paths
    chroma_dir = base_dir / "chroma_db"
    graph_file = base_dir / "knowledge_graph.json"
    output_dir = base_dir / "compressed"

    # Load and filter concepts
    with open(graph_file) as f:
        graph = json.load(f)
    degree = compute_degrees(graph)
    concepts = filter_concepts(graph, MIN_DEGREE)
    concepts.sort(key=lambda c: -(degree.get(c["name"], 0)))

    if end_idx is None:
        end_idx = len(concepts)

    batch = concepts[start_idx:end_idx]
    total_concepts = len(graph["concepts"])
    print(f"Knowledge graph: {total_concepts} concepts, {len(graph['edges'])} edges")
    print(f"Filtered to {len(concepts)} concepts with degree >= {MIN_DEGREE}")
    print(f"[BATCH {start_idx}-{end_idx}] {len(batch)} concepts, model={model}")
    sys.stdout.flush()

    # Setup clients
    openai_client = OpenAI()
    coll_name = collection_name or config.collection_names[0] if config.collection_names else DEFAULT_COLLECTION
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_collection(coll_name)
    print(f"ChromaDB: {collection.count()} chunks in '{coll_name}'")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Stagger start
    stagger = random.uniform(0, 3)
    time.sleep(stagger)

    # Process each concept
    success = 0
    errors = 0
    skipped = 0
    t0 = time.time()

    system_prompt = config.summary_system_prompt if config else ""
    sections = config.summary_sections if config else None

    for i, concept in enumerate(batch):
        idx = start_idx + i
        name = concept["name"]
        slug = slugify(name)
        output_file = output_dir / f"{slug}.md"

        if output_file.exists() and output_file.stat().st_size > 100:
            skipped += 1
            continue

        try:
            query_text = f"{concept.get('display_name', name)}: {concept.get('description', '')}"

            results = with_retry(
                lambda qt=query_text: query_chroma(collection, openai_client, qt)
            )
            passages_text = format_passages(results)
            neighbors = get_neighbors(graph, name)

            summary = with_retry(
                lambda c=concept, pt=passages_text, n=neighbors: generate_summary(
                    openai_client, c, pt, n, model,
                    system_prompt=system_prompt,
                    summary_sections=sections,
                )
            )

            with open(output_file, 'w') as f:
                f.write(summary)

            success += 1
            elapsed = time.time() - t0
            rate = success / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{idx}] OK {name} ({rate:.0f}/min)")
            sys.stdout.flush()

        except Exception as e:
            errors += 1
            print(f"  [{idx}] ERROR {name}: {e}")
            sys.stdout.flush()
            time.sleep(2)

    elapsed = time.time() - t0
    print(f"\n[BATCH {start_idx}-{end_idx}] Done in {elapsed:.0f}s: "
          f"{success} generated, {skipped} skipped, {errors} errors")
    sys.stdout.flush()
