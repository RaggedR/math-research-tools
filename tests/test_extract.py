"""Tests for kg.extract — GPT concept extraction and name normalization."""

import json
from unittest.mock import MagicMock

import pytest


class TestNormalizeName:
    def test_basic_normalization(self):
        from kg.extract import normalize_name

        assert normalize_name("Rogers-Ramanujan identity") == "rogers-ramanujan identities"
        assert normalize_name("Bailey's lemma") == "bailey lemma"
        assert normalize_name("RR identities") == "rogers-ramanujan identities"

    def test_case_insensitive(self):
        from kg.extract import normalize_name

        assert normalize_name("ROGERS-RAMANUJAN IDENTITY") == "rogers-ramanujan identities"
        assert normalize_name("Bailey's Lemma") == "bailey lemma"

    def test_strips_whitespace(self):
        from kg.extract import normalize_name

        assert normalize_name("  Bailey's lemma  ") == "bailey lemma"

    def test_unknown_passes_through(self):
        from kg.extract import normalize_name

        assert normalize_name("some novel concept") == "some novel concept"

    def test_plural_normalizations(self):
        from kg.extract import normalize_name

        assert normalize_name("Hall-Littlewood polynomial") == "hall-littlewood polynomials"
        assert normalize_name("Hall-Littlewood functions") == "hall-littlewood polynomials"
        assert normalize_name("Schur polynomial") == "schur functions"
        assert normalize_name("plane partition") == "plane partitions"

    def test_abbreviations(self):
        from kg.extract import normalize_name

        assert normalize_name("CPP") == "cylindric partitions"
        assert normalize_name("CPPs") == "cylindric partitions"
        assert normalize_name("Gaussian polynomials") == "q-binomial coefficients"


class TestSelectRepresentativeChunks:
    def test_short_list_unchanged(self):
        from kg.extract import select_representative_chunks

        chunks = [{"text": f"chunk {i}"} for i in range(3)]
        result = select_representative_chunks(chunks)
        assert len(result) == 3

    def test_long_list_trimmed(self):
        from kg.extract import select_representative_chunks

        chunks = [{"text": f"chunk {i}"} for i in range(20)]
        result = select_representative_chunks(chunks)
        assert len(result) == 4  # default MAX_CHUNKS_PER_PAPER = 4
        # First 2 + last 2
        assert result[0]["text"] == "chunk 0"
        assert result[1]["text"] == "chunk 1"
        assert result[2]["text"] == "chunk 18"
        assert result[3]["text"] == "chunk 19"

    def test_custom_max_chunks(self):
        from kg.extract import select_representative_chunks

        chunks = [{"text": f"chunk {i}"} for i in range(10)]
        result = select_representative_chunks(chunks, max_chunks=6)
        assert len(result) == 6


class TestExtractConcepts:
    def test_basic_extraction(self, mock_openai_client):
        from kg.extract import extract_concepts

        result = extract_concepts("some text", "paper.pdf", mock_openai_client)
        assert "concepts" in result
        assert "relationships" in result
        assert len(result["concepts"]) >= 1
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_handles_error(self, mock_openai_client):
        from kg.extract import extract_concepts

        mock_openai_client.chat.completions.create.side_effect = Exception("API error")
        result = extract_concepts("some text", "paper.pdf", mock_openai_client)
        assert result == {"concepts": [], "relationships": []}

    def test_uses_correct_model(self, mock_openai_client):
        from kg.extract import extract_concepts

        extract_concepts("text", "paper.pdf", mock_openai_client)
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["response_format"] == {"type": "json_object"}
