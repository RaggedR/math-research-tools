"""Configuration constants for the knowledge graph pipeline."""

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
COLLECTION_NAMES = ["lit_review", "math_papers"]
INGEST_COLLECTION = "lit_review"
MAX_CHUNKS_PER_PAPER = 4  # first 2 + last 2 chunks per paper
MIN_DEGREE_FOR_VIZ = 2

# Plaintext files: split into sections of roughly this many characters
PLAINTEXT_SECTION_SIZE = 3000

EXTRACTION_PROMPT = """You are a mathematical knowledge extractor. Given text from a research paper, extract:

1. **Concepts**: Mathematical objects, theorems, conjectures, techniques, and structures mentioned.
2. **Relationships**: How concepts relate to each other.

Return JSON with this exact schema:
{
  "concepts": [
    {
      "name": "short canonical name (e.g., 'Rogers-Ramanujan identities')",
      "type": "one of: object, theorem, conjecture, technique, identity, formula, person, definition",
      "description": "one sentence description if clear from context"
    }
  ],
  "relationships": [
    {
      "source": "concept name (must match a concept above)",
      "target": "concept name (must match a concept above)",
      "relation": "one of: proves, generalizes, uses, implies, is_instance_of, conjectured_by, related_to, equivalent_to, specializes_to, defined_in",
      "detail": "brief explanation"
    }
  ]
}

Rules:
- Use canonical names (e.g., "Bailey lemma" not "Bailey's lemma" or "the lemma of Bailey")
- Prefer established mathematical names over ad-hoc descriptions
- Extract 3-15 concepts per passage (focus on the most important ones)
- Only include relationships that are explicitly stated or clearly implied
- If the text is mostly formulas with little conceptual content, return fewer items
- For people, only include them if they are credited with a specific result in this passage"""

NORMALIZE = {
    "rogers-ramanujan identity": "rogers-ramanujan identities",
    "rr identities": "rogers-ramanujan identities",
    "rr identity": "rogers-ramanujan identities",
    "bailey's lemma": "bailey lemma",
    "hall-littlewood polynomial": "hall-littlewood polynomials",
    "hall-littlewood function": "hall-littlewood polynomials",
    "hall-littlewood functions": "hall-littlewood polynomials",
    "macdonald polynomial": "macdonald polynomials",
    "schur function": "schur functions",
    "schur polynomial": "schur functions",
    "cylindric plane partition": "cylindric partitions",
    "cylindric plane partitions": "cylindric partitions",
    "cpp": "cylindric partitions",
    "cpps": "cylindric partitions",
    "q-binomial coefficient": "q-binomial coefficients",
    "gaussian polynomial": "q-binomial coefficients",
    "gaussian polynomials": "q-binomial coefficients",
    "quasi-symmetric function": "quasi-symmetric functions",
    "quasisymmetric function": "quasi-symmetric functions",
    "quasisymmetric functions": "quasi-symmetric functions",
    "crystal base": "crystal bases",
    "andrews-gordon identity": "andrews-gordon identities",
    "a2 andrews-gordon identities": "a2 andrews-gordon identities",
    "plane partition": "plane partitions",
}

TYPE_COLORS = {
    "object": "#4A90D9",
    "theorem": "#E74C3C",
    "conjecture": "#F39C12",
    "technique": "#2ECC71",
    "identity": "#9B59B6",
    "formula": "#1ABC9C",
    "person": "#E67E22",
    "definition": "#3498DB",
}

# Supported file extensions for text extraction
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text", ".markdown"}
