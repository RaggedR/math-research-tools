"""Literature agent with progressive drill-down search.

Searches Level 2 (meta-summaries) -> Level 1 (concept summaries) -> Level 0 (raw papers)
and synthesizes a compact answer with citations.

Usage as a library::

    from kg.agent import run_agent

    result = run_agent(
        question="What is the spectral gap for random regular graphs?",
        base_dir="/path/to/data",
    )
    print(result["answer"])
"""

import logging
from pathlib import Path

from .config import DomainConfig, EMBEDDING_MODEL, load_config

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL  # "text-embedding-3-small"
DEFAULT_ANSWER_MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.35

DEFAULT_COLLECTIONS = {
    0: "lit_review",
    1: "level1_summaries",
}


# ── Path helpers ─────────────────────────────────────────────────────

def get_agent_paths(base_dir):
    """Derive all agent-relevant paths from a base data directory.

    Returns:
        dict with keys: chroma_dir, meta_dir, compressed_dir
    """
    d = Path(base_dir)
    return {
        "chroma_dir": d / "chroma_db",
        "meta_dir": d / "meta-summaries",
        "compressed_dir": d / "compressed",
    }


# ── Utilities ────────────────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0


# ── Search functions ─────────────────────────────────────────────────

def search_level2(question, openai_client, meta_dir, embedding_model=None):
    """Search Level 2 meta-summaries by embedding similarity.

    Args:
        question: The query string.
        openai_client: An OpenAI client instance.
        meta_dir: Path to the meta-summaries directory.
        embedding_model: Embedding model name (default: text-embedding-3-small).

    Returns:
        (passages, scored) where passages is a list of passage dicts or None,
        and scored is a list of (similarity, name, text) tuples.
    """
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    meta_dir = Path(meta_dir)
    meta_files = sorted(meta_dir.glob("*.md"))

    if not meta_files:
        return None, []

    q_response = openai_client.embeddings.create(
        model=embedding_model, input=[question[:8000]]
    )
    q_emb = q_response.data[0].embedding

    texts = []
    names = []
    for mf in meta_files:
        content = mf.read_text()[:2000]
        texts.append(content)
        names.append(mf.stem)

    t_response = openai_client.embeddings.create(
        model=embedding_model, input=texts
    )
    t_embs = [e.embedding for e in t_response.data]

    scored = []
    for i, (name, emb) in enumerate(zip(names, t_embs)):
        sim = cosine_sim(q_emb, emb)
        scored.append((sim, name, texts[i]))

    scored.sort(key=lambda x: -x[0])
    top = scored[:3]

    if top[0][0] < SIMILARITY_THRESHOLD:
        return None, top

    passages = []
    for sim, name, text in top:
        full_text = (meta_dir / f"{name}.md").read_text()
        passages.append({
            "level": 2,
            "source": f"meta-summaries/{name}.md",
            "similarity": sim,
            "text": full_text[:4000],
        })

    return passages, top


def search_level(question, level, openai_client, chroma_client,
                 collections=None, embedding_model=None, top_k=8):
    """Search a ChromaDB collection at the given hierarchy level.

    Args:
        question: The query string.
        level: Hierarchy level (0 = raw papers, 1 = concept summaries).
        openai_client: An OpenAI client instance.
        chroma_client: A ChromaDB client instance.
        collections: Mapping of level -> collection name. Defaults to
            DEFAULT_COLLECTIONS if not provided.
        embedding_model: Embedding model name (default: text-embedding-3-small).
        top_k: Number of results to retrieve.

    Returns:
        List of passage dicts, or None if the collection is not found.
    """
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    collections = collections or DEFAULT_COLLECTIONS
    collection_name = collections.get(level)

    if collection_name is None:
        logger.warning("No collection configured for level %d", level)
        return None

    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        logger.warning("Collection '%s' not found", collection_name)
        return None

    response = openai_client.embeddings.create(
        model=embedding_model, input=[question[:8000]]
    )
    q_emb = response.data[0].embedding

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        sim = 1 - dist
        passages.append({
            "level": level,
            "source": meta.get("source", "unknown"),
            "concept": meta.get("concept", ""),
            "similarity": sim,
            "text": doc.strip()[:2000],
        })

    return passages


# ── Synthesis ────────────────────────────────────────────────────────

def synthesize_answer(question, all_passages, openai_client,
                      answer_model=None, domain_name=None):
    """Synthesize an answer from retrieved passages using an LLM.

    Args:
        question: The original question.
        all_passages: List of passage dicts from search functions.
        openai_client: An OpenAI client instance.
        answer_model: Chat model name (default: gpt-4o-mini).
        domain_name: Domain name for prompt context (e.g. "mathematical",
            "evolutionary computation"). If None, uses generic wording.

    Returns:
        The synthesized answer as a string.
    """
    answer_model = answer_model or DEFAULT_ANSWER_MODEL
    domain = domain_name or "research"

    formatted = []
    for p in all_passages:
        level_name = {0: "Raw Paper", 1: "Concept Summary", 2: "Meta-Summary"}[
            p["level"]
        ]
        formatted.append(
            f"[{level_name}: {p['source']} (similarity: {p['similarity']:.3f})]\n"
            f"{p['text']}"
        )

    passages_text = "\n\n---\n\n".join(formatted)

    prompt = f"""You are a {domain} research assistant answering questions \
using a hierarchical literature database.

QUESTION: {question}

Below are passages retrieved from three levels of the literature hierarchy:
- Level 2 (meta-summaries): Bird's-eye overviews of research areas
- Level 1 (concept summaries): Focused summaries of individual concepts
- Level 0 (raw papers): Direct excerpts from research papers

INSTRUCTIONS:
- Answer the question concisely but precisely (aim for 5-15 sentences).
- Use LaTeX notation for mathematics where appropriate ($...$, $$...$$).
- Cite sources using [Author, Year] or [concept-name] format.
- If the passages don't contain enough information to answer fully, say so.
- Prioritize precision over coverage — it's better to give a focused answer \
than a vague one.
- End with a "Sources:" section listing the most relevant sources.

RETRIEVED PASSAGES:
{passages_text}"""

    response = openai_client.chat.completions.create(
        model=answer_model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a precise {domain} research assistant. "
                    "Cite sources."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0.2,
    )

    return response.choices[0].message.content


# ── Progressive search ───────────────────────────────────────────────

def progressive_search(question, openai_client, chroma_client, meta_dir,
                       collections=None, embedding_model=None,
                       start_level=2, deep=False):
    """Search progressively from Level 2 down to Level 0.

    Starts at the highest level (meta-summaries) and drills down if the
    similarity is low or ``deep=True``.

    Args:
        question: The query string.
        openai_client: An OpenAI client instance.
        chroma_client: A ChromaDB client instance.
        meta_dir: Path to the meta-summaries directory.
        collections: Mapping of level -> collection name.
        embedding_model: Embedding model name.
        start_level: Highest level to search (default 2).
        deep: If True, always search all levels.

    Returns:
        (all_passages, levels_searched) tuple.
    """
    all_passages = []
    levels_searched = []

    # Level 2: meta-summaries (file-based)
    if start_level >= 2:
        logger.info("Searching Level 2 (meta-summaries)...")
        l2_passages, l2_scores = search_level2(
            question, openai_client, meta_dir,
            embedding_model=embedding_model,
        )
        if l2_passages:
            all_passages.extend(l2_passages)
            levels_searched.append(2)
            best_sim = l2_scores[0][0] if l2_scores else 0
            logger.info(
                "  Found %d matches (best similarity: %.3f)",
                len(l2_passages), best_sim,
            )
            if best_sim > 0.5 and not deep:
                return all_passages, levels_searched
        else:
            logger.info("  No strong matches at Level 2")

    # Level 1: concept summaries (ChromaDB)
    if start_level >= 1 or not all_passages or deep:
        logger.info("Searching Level 1 (concept summaries)...")
        l1_passages = search_level(
            question, 1, openai_client, chroma_client,
            collections=collections, embedding_model=embedding_model,
            top_k=10,
        )
        if l1_passages:
            all_passages.extend(l1_passages)
            levels_searched.append(1)
            best_sim = max(p["similarity"] for p in l1_passages)
            logger.info(
                "  Found %d matches (best similarity: %.3f)",
                len(l1_passages), best_sim,
            )
            if best_sim > 0.5 and not deep:
                return all_passages, levels_searched

    # Level 0: raw papers (ChromaDB)
    if start_level == 0 or not all_passages or deep:
        logger.info("Searching Level 0 (raw papers)...")
        l0_passages = search_level(
            question, 0, openai_client, chroma_client,
            collections=collections, embedding_model=embedding_model,
            top_k=8,
        )
        if l0_passages:
            all_passages.extend(l0_passages)
            levels_searched.append(0)
            best_sim = max(p["similarity"] for p in l0_passages)
            logger.info(
                "  Found %d matches (best similarity: %.3f)",
                len(l0_passages), best_sim,
            )

    return all_passages, levels_searched


# ── Main entry point ─────────────────────────────────────────────────

def run_agent(question, base_dir, config=None, start_level=2, deep=False,
              answer_model=None, embedding_model=None):
    """Run the literature agent: search, then synthesize an answer.

    This is the primary entry point for library consumers.

    Args:
        question: The research question to answer.
        base_dir: Root data directory containing chroma_db/, meta-summaries/,
            and compressed/ sub-directories.
        config: Optional DomainConfig instance. If None, one is loaded from
            base_dir via the standard config resolution chain.
        start_level: Highest hierarchy level to begin searching (default 2).
        deep: If True, search all levels regardless of similarity scores.
        answer_model: Override the default chat model for synthesis.
        embedding_model: Override the default embedding model.

    Returns:
        dict with keys:
            answer (str): The synthesized answer text.
            levels_searched (list[int]): Which levels were consulted.
            num_passages (int): Total passages retrieved.
    """
    import chromadb
    from openai import OpenAI

    base_dir = Path(base_dir)
    paths = get_agent_paths(base_dir)

    # Resolve config
    if config is None:
        config = load_config(data_dir=base_dir)

    # Build collection name mapping from config
    collections = dict(DEFAULT_COLLECTIONS)
    if config and config.collection_names:
        # First collection name maps to level 0 (raw papers)
        collections[0] = config.collection_names[0]
        # Second collection name maps to level 1, if available
        if len(config.collection_names) > 1:
            collections[1] = config.collection_names[1]

    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(paths["chroma_dir"]))

    # Search
    all_passages, levels_searched = progressive_search(
        question,
        openai_client,
        chroma_client,
        meta_dir=paths["meta_dir"],
        collections=collections,
        embedding_model=embedding_model,
        start_level=start_level,
        deep=deep,
    )

    if not all_passages:
        return {
            "answer": "No relevant passages found in the literature database.",
            "levels_searched": levels_searched,
            "num_passages": 0,
        }

    # Synthesize
    domain_name = config.name if config else None
    answer = synthesize_answer(
        question, all_passages, openai_client,
        answer_model=answer_model,
        domain_name=domain_name,
    )

    return {
        "answer": answer,
        "levels_searched": levels_searched,
        "num_passages": len(all_passages),
    }
