"""Configuration constants and domain config loading for the knowledge graph pipeline.

Three-tier config resolution:
  1. Explicit --config path (highest priority)
  2. domain.yaml in the data directory (points to a configs/ YAML)
  3. Fallback to math.yaml (default)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# ── Package-level constants (unchanged from original) ─────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_PAPER = 4  # first 2 + last 2 chunks per paper
MIN_DEGREE_FOR_VIZ = 2
PLAINTEXT_SECTION_SIZE = 3000
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text", ".markdown"}

# ── Paths ──────────────────────────────────────────────────────────────

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# ── Domain configuration ──────────────────────────────────────────────


@dataclass
class DomainConfig:
    """Domain-specific configuration loaded from YAML."""

    name: str = "math"
    extraction_prompt: str = ""
    normalize: dict = field(default_factory=dict)
    type_colors: dict = field(default_factory=dict)
    collection_names: list = field(default_factory=lambda: ["lit_review"])
    ingest_collection: str = "lit_review"
    summary_system_prompt: str = ""
    concept_types: list = field(default_factory=list)
    relationship_types: list = field(default_factory=list)


def _load_yaml(path):
    """Load and return a YAML file as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_config(raw):
    """Build a DomainConfig from a raw dict, ignoring unknown keys."""
    known_fields = {f.name for f in DomainConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    return DomainConfig(**filtered)


def load_config(config_path=None, data_dir=None):
    """Load domain configuration with three-tier resolution.

    Priority:
      1. config_path (explicit path to a YAML file)
      2. domain.yaml in data_dir (contains "config: <name>" pointer)
      3. Fallback to configs/math.yaml

    A domain.yaml can also contain field-level overrides that are merged
    on top of the base config it points to.

    Args:
        config_path: Explicit path to a domain config YAML file.
        data_dir: Data directory that may contain a domain.yaml.

    Returns:
        DomainConfig instance.
    """
    # Tier 1: explicit config path
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        raw = _load_yaml(path)
        return _build_config(raw)

    # Tier 2: domain.yaml in data directory
    if data_dir is not None:
        domain_yaml = Path(data_dir) / "domain.yaml"
        if domain_yaml.exists():
            pointer = _load_yaml(domain_yaml)
            config_name = pointer.pop("config", None)
            if config_name:
                base_path = CONFIGS_DIR / f"{config_name}.yaml"
                if base_path.exists():
                    raw = _load_yaml(base_path)
                    # Apply overrides from domain.yaml
                    raw.update(pointer)
                    return _build_config(raw)

    # Tier 3: fallback to math config
    math_path = CONFIGS_DIR / "math.yaml"
    if math_path.exists():
        raw = _load_yaml(math_path)
        return _build_config(raw)

    # Ultimate fallback: bare defaults
    return DomainConfig()


# ── Backwards-compatible module-level constants ───────────────────────
# These load the default math config so existing code that does
# `from kg.config import EXTRACTION_PROMPT` continues to work.

_default = load_config()
EXTRACTION_PROMPT = _default.extraction_prompt
NORMALIZE = _default.normalize
TYPE_COLORS = _default.type_colors
COLLECTION_NAMES = _default.collection_names
INGEST_COLLECTION = _default.ingest_collection
