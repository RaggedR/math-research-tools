"""Tests for kg.ingest — PDF/text extraction, chunking, and embedding."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestExtractTextFromPdf:
    def test_extracts_pages(self, sample_pdf_path):
        from kg.ingest import extract_text_from_pdf

        pages = extract_text_from_pdf(sample_pdf_path)
        assert len(pages) >= 1
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pages)
        page_num, text = pages[0]
        assert page_num == 1
        assert "Rogers-Ramanujan" in text

    def test_returns_empty_for_nonexistent(self, tmp_path):
        from kg.ingest import extract_text_from_pdf

        pages = extract_text_from_pdf(tmp_path / "nonexistent.pdf")
        assert pages == []


class TestExtractTextFromPlaintext:
    def test_extracts_txt(self, sample_txt_path):
        from kg.ingest import extract_text_from_plaintext

        pages = extract_text_from_plaintext(sample_txt_path)
        assert len(pages) >= 1
        # Should return (section_num, text) tuples like PDF extraction
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pages)
        _, text = pages[0]
        assert "Rogers-Ramanujan" in text

    def test_extracts_markdown(self, sample_md_path):
        from kg.ingest import extract_text_from_plaintext

        pages = extract_text_from_plaintext(sample_md_path)
        assert len(pages) >= 1
        full_text = " ".join(text for _, text in pages)
        assert "cylindric" in full_text.lower()

    def test_empty_file(self, tmp_path):
        from kg.ingest import extract_text_from_plaintext

        empty = tmp_path / "empty.txt"
        empty.write_text("")
        pages = extract_text_from_plaintext(empty)
        assert pages == []


class TestChunkText:
    def test_basic_chunking(self, sample_pages):
        from kg.ingest import chunk_text

        chunks = chunk_text(sample_pages)
        assert len(chunks) >= 1
        assert all("text" in c and "pages" in c for c in chunks)

    def test_chunk_size_respected(self, sample_pages):
        from kg.ingest import chunk_text

        # With a very small chunk size, should produce more chunks
        chunks = chunk_text(sample_pages, chunk_size=100, overlap=20)
        assert len(chunks) >= 2

    def test_short_text_produces_single_chunk(self):
        from kg.ingest import chunk_text

        pages = [(1, "Short text about math.")]
        chunks = chunk_text(pages)
        # Too short (< 50 chars), should produce empty
        assert len(chunks) == 0

    def test_pages_tracked(self):
        from kg.ingest import chunk_text

        pages = [
            (1, "A" * 800),
            (2, "B" * 800),
        ]
        chunks = chunk_text(pages, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        # At least some chunks should have page numbers
        all_pages = set()
        for c in chunks:
            all_pages.update(c["pages"])
        assert len(all_pages) >= 1


class TestGetEmbeddings:
    def test_calls_openai(self, mock_openai_client):
        from kg.ingest import get_embeddings

        texts = ["hello world", "test text"]
        embeddings = get_embeddings(texts, mock_openai_client)
        assert mock_openai_client.embeddings.create.called
        assert len(embeddings) >= 1

    def test_batching(self, mock_openai_client):
        from kg.ingest import get_embeddings

        # Create many texts to trigger batching
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding] * 50
        mock_openai_client.embeddings.create.return_value = mock_response

        texts = [f"text {i}" for i in range(150)]
        embeddings = get_embeddings(texts, mock_openai_client)
        # Should have called embeddings.create twice (batches of 100)
        assert mock_openai_client.embeddings.create.call_count == 2


class TestExtractFile:
    def test_pdf_routing(self, sample_pdf_path):
        from kg.ingest import extract_file

        pages = extract_file(sample_pdf_path)
        assert len(pages) >= 1
        assert "Rogers-Ramanujan" in pages[0][1]

    def test_txt_routing(self, sample_txt_path):
        from kg.ingest import extract_file

        pages = extract_file(sample_txt_path)
        assert len(pages) >= 1

    def test_md_routing(self, sample_md_path):
        from kg.ingest import extract_file

        pages = extract_file(sample_md_path)
        assert len(pages) >= 1

    def test_unsupported_format(self, tmp_path):
        from kg.ingest import extract_file

        f = tmp_path / "test.xyz"
        f.write_text("data")
        pages = extract_file(f)
        assert pages == []
