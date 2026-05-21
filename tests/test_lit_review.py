"""Tests for bin/lit_review.py — OpenAlex source and abstract reconstruction.

These tests mock all HTTP calls; no network access occurs.
"""

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the bin/ directory importable so we can import lit_review directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
import lit_review


# ---------------------------------------------------------------------------
# _openalex_abstract — inverted-index reconstruction
# ---------------------------------------------------------------------------

class TestOpenAlexAbstract:
    """Unit tests for the abstract_inverted_index → plain text helper."""

    def test_basic_reconstruction(self):
        inverted = {"The": [0], "quick": [1], "brown": [2], "fox": [3]}
        result = lit_review._openalex_abstract(inverted)
        assert result == "The quick brown fox"

    def test_multi_occurrence_word(self):
        # "the" appears at positions 0 and 3
        inverted = {"the": [0, 3], "cat": [1], "sat": [2]}
        result = lit_review._openalex_abstract(inverted)
        assert result == "the cat sat the"

    def test_none_returns_empty(self):
        assert lit_review._openalex_abstract(None) == ''

    def test_empty_dict_returns_empty(self):
        assert lit_review._openalex_abstract({}) == ''

    def test_preserves_word_order(self):
        # Positions are non-contiguous but should still be sorted correctly
        inverted = {"C": [2], "A": [0], "B": [1]}
        result = lit_review._openalex_abstract(inverted)
        assert result == "A B C"

    def test_realistic_chemistry_abstract(self):
        inverted = {
            "Enzyme": [0],
            "immobilisation": [1],
            "within": [2],
            "metal-organic": [3],
            "frameworks": [4],
            "enhances": [5],
            "stability.": [6],
        }
        result = lit_review._openalex_abstract(inverted)
        assert result == "Enzyme immobilisation within metal-organic frameworks enhances stability."


# ---------------------------------------------------------------------------
# _parse_openalex_work — dict-shape contract
# ---------------------------------------------------------------------------

class TestParseOpenAlexWork:
    """Verify that _parse_openalex_work returns the expected dict shape."""

    def _sample_work(self, **overrides):
        """Minimal valid OpenAlex work object."""
        base = {
            "id": "https://openalex.org/W1234567890",
            "title": "HOF enzyme immobilisation study",
            "publication_year": 2023,
            "doi": "https://doi.org/10.1021/jacs.3c01234",
            "abstract_inverted_index": {
                "Porous": [0], "frameworks": [1], "encapsulate": [2], "enzymes.": [3]
            },
            "authorships": [
                {"author": {"display_name": "Robin Langer"}},
                {"author": {"display_name": "Jane Smith"}},
            ],
            "best_oa_location": None,
        }
        base.update(overrides)
        return base

    def test_returns_expected_keys(self):
        work = self._sample_work()
        result = lit_review._parse_openalex_work(work)
        assert result is not None
        for key in ("base_id", "title", "abstract", "authors", "published",
                    "pdf_url", "doi", "source"):
            assert key in result, f"Missing key: {key}"

    def test_source_is_openalex(self):
        result = lit_review._parse_openalex_work(self._sample_work())
        assert result["source"] == "openalex"

    def test_doi_used_as_base_id(self):
        result = lit_review._parse_openalex_work(self._sample_work())
        # DOI: 10.1021/jacs.3c01234 → base_id encodes it
        assert result["base_id"].startswith("doi_")
        assert "10_1021" in result["base_id"]

    def test_openalex_id_fallback_when_no_doi(self):
        work = self._sample_work(doi=None)
        result = lit_review._parse_openalex_work(work)
        assert result is not None
        assert result["base_id"].startswith("openalex_W")

    def test_abstract_reconstructed(self):
        result = lit_review._parse_openalex_work(self._sample_work())
        assert "Porous" in result["abstract"]
        assert "enzymes." in result["abstract"]

    def test_returns_none_when_no_abstract(self):
        work = self._sample_work(abstract_inverted_index=None)
        assert lit_review._parse_openalex_work(work) is None

    def test_returns_none_when_empty_abstract(self):
        work = self._sample_work(abstract_inverted_index={})
        assert lit_review._parse_openalex_work(work) is None

    def test_returns_none_when_no_title(self):
        work = self._sample_work(title=None)
        assert lit_review._parse_openalex_work(work) is None

    def test_returns_none_when_no_id_and_no_doi(self):
        work = self._sample_work(id=None, doi=None)
        assert lit_review._parse_openalex_work(work) is None

    def test_authors_extracted(self):
        result = lit_review._parse_openalex_work(self._sample_work())
        assert result["authors"] == ["Robin Langer", "Jane Smith"]

    def test_published_year(self):
        result = lit_review._parse_openalex_work(self._sample_work())
        assert result["published"] == "2023"

    def test_pdf_url_none_for_paywalled(self):
        work = self._sample_work(best_oa_location=None)
        result = lit_review._parse_openalex_work(work)
        assert result["pdf_url"] is None

    def test_pdf_url_from_oa_location(self):
        work = self._sample_work(
            best_oa_location={"pdf_url": "https://example.com/paper.pdf"}
        )
        result = lit_review._parse_openalex_work(work)
        assert result["pdf_url"] == "https://example.com/paper.pdf"


# ---------------------------------------------------------------------------
# search_openalex — mocked HTTP
# ---------------------------------------------------------------------------

def _make_openalex_response(works):
    """Build a minimal OpenAlex /works JSON response."""
    payload = json.dumps({"results": works}).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _sample_work_fixture():
    return {
        "id": "https://openalex.org/W9999999999",
        "title": "Vibronic coupling in laccase immobilised on MOF",
        "publication_year": 2024,
        "doi": "https://doi.org/10.1039/d4sc01234a",
        "abstract_inverted_index": {
            "Laccase": [0], "immobilised": [1], "on": [2],
            "MOF": [3], "shows": [4], "vibronic": [5], "coupling.": [6],
        },
        "authorships": [{"author": {"display_name": "A. Chemist"}}],
        "best_oa_location": None,
    }


class TestSearchOpenAlex:
    """Integration-level tests for search_openalex with mocked urllib."""

    @patch("urllib.request.urlopen")
    def test_returns_list_of_dicts(self, mock_urlopen):
        mock_urlopen.return_value = _make_openalex_response([_sample_work_fixture()])
        results = lit_review.search_openalex("laccase MOF vibronic", max_results=1)
        assert isinstance(results, list)
        assert len(results) == 1

    @patch("urllib.request.urlopen")
    def test_dict_shape_contract(self, mock_urlopen):
        mock_urlopen.return_value = _make_openalex_response([_sample_work_fixture()])
        results = lit_review.search_openalex("laccase MOF", max_results=1)
        paper = results[0]
        for key in ("base_id", "title", "abstract", "authors", "published",
                    "pdf_url", "doi", "source"):
            assert key in paper, f"Missing key in search result: {key}"
        assert paper["source"] == "openalex"

    @patch("urllib.request.urlopen")
    def test_skips_works_without_abstract(self, mock_urlopen):
        no_abstract = dict(_sample_work_fixture())
        no_abstract["abstract_inverted_index"] = None
        mock_urlopen.return_value = _make_openalex_response([no_abstract])
        results = lit_review.search_openalex("laccase MOF", max_results=5)
        assert results == []

    @patch("urllib.request.urlopen")
    def test_empty_results_returns_empty_list(self, mock_urlopen):
        mock_urlopen.return_value = _make_openalex_response([])
        results = lit_review.search_openalex("no results query", max_results=10)
        assert results == []

    @patch("urllib.request.urlopen")
    def test_network_error_returns_empty_list(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        results = lit_review.search_openalex("any query", max_results=10)
        assert results == []

    @patch("urllib.request.urlopen")
    def test_mailto_in_request_url(self, mock_urlopen):
        """Polite-pool email is included in the API request URL (URL-encoded form)."""
        mock_urlopen.return_value = _make_openalex_response([_sample_work_fixture()])
        lit_review.search_openalex("enzyme framework", max_results=1)
        call_args = mock_urlopen.call_args
        # urlopen receives a Request object; get its full_url.
        # urllib.parse.urlencode encodes '@' as '%40', so check for both forms.
        request_obj = call_args[0][0]
        url = request_obj.full_url
        assert "langer.robin" in url and ("gmail.com" in url or "gmail" in url)
