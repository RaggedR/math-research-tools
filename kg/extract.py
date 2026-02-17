"""GPT-4o-mini concept extraction and name normalization."""

import json
import logging

from .config import EXTRACTION_PROMPT, MAX_CHUNKS_PER_PAPER, NORMALIZE

logger = logging.getLogger(__name__)


def normalize_name(name):
    """Normalize concept names for deduplication.

    Lowercases, strips whitespace, and applies the synonym map from config.
    """
    name = name.strip().lower()
    return NORMALIZE.get(name, name)


def select_representative_chunks(chunks, max_chunks=MAX_CHUNKS_PER_PAPER):
    """Pick first + last chunks from a paper (intro + conclusion).

    For a paper with many chunks, returns the first half and last half
    of max_chunks (e.g., first 2 + last 2 for max_chunks=4).
    """
    if len(chunks) <= max_chunks:
        return chunks
    half = max_chunks // 2
    return chunks[:half] + chunks[-half:]


def extract_concepts(text, paper_name, client):
    """Use GPT-4o-mini to extract concepts and relationships from text.

    Returns dict with "concepts" and "relationships" keys.
    On error, returns empty lists for both.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Paper: {paper_name}\n\n{text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("Error extracting from %s: %s", paper_name, e)
        return {"concepts": [], "relationships": []}
