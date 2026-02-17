"""kg — Knowledge graph library for extracting concepts from research papers."""

from .config import TYPE_COLORS, NORMALIZE
from .extract import extract_concepts, normalize_name, select_representative_chunks
from .graph import build_graph, merge_extractions, prepare_viz_data
from .ingest import (
    chunk_text,
    extract_file,
    extract_text_from_pdf,
    extract_text_from_plaintext,
    get_embeddings,
    ingest_files,
)
from .visualize import generate_html

__all__ = [
    "TYPE_COLORS",
    "NORMALIZE",
    "extract_concepts",
    "normalize_name",
    "select_representative_chunks",
    "build_graph",
    "merge_extractions",
    "prepare_viz_data",
    "chunk_text",
    "extract_file",
    "extract_text_from_pdf",
    "extract_text_from_plaintext",
    "get_embeddings",
    "ingest_files",
    "generate_html",
]
