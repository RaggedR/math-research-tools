"""GPT-4o-mini concept extraction and name normalization."""

import json
import logging

from .config import EXTRACTION_PROMPT, MAX_CHUNKS_PER_PAPER, NORMALIZE

logger = logging.getLogger(__name__)


def normalize_name(name, normalize_table=None):
    """Normalize concept names for deduplication.

    Lowercases, strips whitespace, and applies the synonym map.

    Args:
        name: Raw concept name.
        normalize_table: Optional dict of synonym mappings. If None,
            uses the default NORMALIZE table from config.
    """
    table = normalize_table if normalize_table is not None else NORMALIZE
    name = name.strip().lower()
    return table.get(name, name)


def select_representative_chunks(chunks, max_chunks=MAX_CHUNKS_PER_PAPER):
    """Pick first + last chunks from a paper (intro + conclusion).

    For a paper with many chunks, returns the first half and last half
    of max_chunks (e.g., first 2 + last 2 for max_chunks=4).
    """
    if len(chunks) <= max_chunks:
        return chunks
    half = max_chunks // 2
    return chunks[:half] + chunks[-half:]


def extract_concepts(text, paper_name, client, extraction_prompt=None):
    """Use GPT-4o-mini to extract concepts and relationships from text.

    Args:
        text: The paper text to extract from.
        paper_name: Name of the paper (for logging).
        client: OpenAI client instance.
        extraction_prompt: Optional custom extraction prompt. If None,
            uses the default EXTRACTION_PROMPT from config.

    Returns dict with "concepts" and "relationships" keys.
    On error, returns empty lists for both.
    """
    prompt = extraction_prompt if extraction_prompt is not None else EXTRACTION_PROMPT
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
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
