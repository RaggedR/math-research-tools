"""Ingest PDFs and text files: extract text, chunk, embed, store in ChromaDB."""

import hashlib
import json
import logging
from pathlib import Path

import chromadb
import fitz  # PyMuPDF

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INGEST_COLLECTION,
    PLAINTEXT_SECTION_SIZE,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF.

    Returns list of (page_num, text) tuples.
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append((page_num + 1, text))
        doc.close()
        return pages
    except Exception as e:
        logger.warning("Could not extract text from %s: %s", pdf_path.name, e)
        return []


def extract_text_from_plaintext(file_path):
    """Extract text from a plaintext or markdown file.

    Returns list of (section_num, text) tuples matching the PDF format,
    so chunk_text() and everything downstream works unchanged.
    """
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("Could not read %s: %s", file_path, e)
        return []

    text = text.strip()
    if not text:
        return []

    # Split into sections of roughly PLAINTEXT_SECTION_SIZE characters,
    # breaking at paragraph boundaries
    sections = []
    section_num = 1
    start = 0
    while start < len(text):
        end = start + PLAINTEXT_SECTION_SIZE
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                sections.append((section_num, chunk))
            break

        # Find a paragraph break near the target size
        para_break = text.rfind("\n\n", start + PLAINTEXT_SECTION_SIZE // 2, end + 500)
        if para_break > start:
            end = para_break

        chunk = text[start:end].strip()
        if chunk:
            sections.append((section_num, chunk))
            section_num += 1
        start = end

    return sections


def extract_file(file_path):
    """Extract text from a file based on its extension.

    Returns list of (section_num, text) tuples.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in {".txt", ".md", ".text", ".markdown"}:
        return extract_text_from_plaintext(path)
    else:
        logger.warning("Unsupported file type: %s", ext)
        return []


def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split extracted pages into overlapping chunks.

    Args:
        pages: list of (page_num, text) tuples
        chunk_size: target chunk size in characters
        overlap: overlap between consecutive chunks
    """
    chunks = []
    current_text = ""
    current_pages = set()

    for page_num, text in pages:
        text = text.replace("\x00", "").strip()
        if not text:
            continue
        current_text += f"\n[Page {page_num}]\n{text}"
        current_pages.add(page_num)

        while len(current_text) >= chunk_size:
            break_at = chunk_size
            para_break = current_text.rfind("\n\n", overlap, chunk_size)
            if para_break > overlap:
                break_at = para_break
            else:
                for sep in [". ", ".\n", ";\n"]:
                    sent_break = current_text.rfind(sep, overlap, chunk_size)
                    if sent_break > overlap:
                        break_at = sent_break + len(sep)
                        break

            chunk = current_text[:break_at].strip()
            if len(chunk) > 50:
                chunks.append({"text": chunk, "pages": sorted(current_pages)})
            current_text = current_text[break_at - overlap :]
            current_pages = set()
            for pn, _ in pages:
                if f"[Page {pn}]" in current_text:
                    current_pages.add(pn)

    if current_text.strip() and len(current_text.strip()) > 50:
        chunks.append({"text": current_text.strip(), "pages": sorted(current_pages)})

    return chunks


def get_embeddings(texts, openai_client):
    """Get embeddings from OpenAI in batches of 100."""
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = [t[:8000] for t in texts[i : i + batch_size]]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])
    return all_embeddings


def ingest_files(file_paths, chroma_dir, openai_client, metadata_map=None,
                 on_progress=None):
    """Ingest files into ChromaDB.

    Args:
        file_paths: list of Path objects to ingest
        chroma_dir: Path to ChromaDB storage directory
        openai_client: OpenAI client instance
        metadata_map: optional dict mapping filename -> paper metadata
        on_progress: optional callback(stage, detail, percent) for progress updates

    Returns:
        Number of chunks stored.
    """
    if metadata_map is None:
        metadata_map = {}

    all_chunks = []
    all_ids = []
    all_metadatas = []
    total_files = len(file_paths)

    for idx, file_path in enumerate(file_paths):
        file_path = Path(file_path)
        if on_progress:
            pct = (idx / total_files) * 50  # ingestion is first 50% of this stage
            on_progress("ingesting", f"{file_path.name} ({idx + 1}/{total_files})", pct)

        pages = extract_file(file_path)
        if not pages:
            continue

        chunks = chunk_text(pages)
        meta = metadata_map.get(file_path.name, {})
        file_id = meta.get("arxiv_id", file_path.stem)
        title = meta.get("title", file_path.stem)

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{file_id}:{i}".encode()).hexdigest()
            all_chunks.append(chunk["text"])
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": file_path.name,
                "arxiv_id": file_id,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    if not all_chunks:
        logger.warning("No text extracted from any file!")
        return 0

    logger.info("%d chunks from %d files", len(all_chunks), total_files)

    if on_progress:
        on_progress("ingesting", "Generating embeddings...", 50)

    embeddings = get_embeddings(all_chunks, openai_client)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_or_create_collection(
        name=INGEST_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        j = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[i:j],
            documents=all_chunks[i:j],
            embeddings=embeddings[i:j],
            metadatas=all_metadatas[i:j],
        )

    if on_progress:
        on_progress("ingesting", "Complete", 100)

    count = collection.count()
    logger.info("Stored %d chunks in %s", count, chroma_dir)
    return count
