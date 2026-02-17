"""Shared test fixtures for the knowledge graph test suite."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf_path():
    return FIXTURES_DIR / "sample.pdf"


@pytest.fixture
def sample_txt_path():
    return FIXTURES_DIR / "sample.txt"


@pytest.fixture
def sample_md_path():
    return FIXTURES_DIR / "sample.md"


@pytest.fixture
def sample_pages():
    """Simulated output of extract_text_from_pdf: list of (page_num, text) tuples."""
    return [
        (1, "Rogers-Ramanujan identities are fundamental results in partition theory. "
            "The Bailey lemma provides a systematic framework for proving these identities. "
            "Andrews-Gordon identities generalize the Rogers-Ramanujan identities."),
        (2, "Schur functions connect to Hall-Littlewood polynomials via specialization. "
            "Macdonald polynomials generalize both. Crystal bases provide a combinatorial "
            "framework for studying representations of quantum groups."),
    ]


@pytest.fixture
def sample_extraction():
    """Sample GPT extraction result for one paper."""
    return {
        "concepts": [
            {"name": "Rogers-Ramanujan identities", "type": "identity",
             "description": "Fundamental partition identities"},
            {"name": "Bailey lemma", "type": "theorem",
             "description": "Framework for proving RR-type identities"},
            {"name": "Andrews-Gordon identities", "type": "identity",
             "description": "Generalization of Rogers-Ramanujan identities"},
        ],
        "relationships": [
            {"source": "Bailey lemma", "target": "Rogers-Ramanujan identities",
             "relation": "proves", "detail": "Provides systematic proof"},
            {"source": "Andrews-Gordon identities", "target": "Rogers-Ramanujan identities",
             "relation": "generalizes", "detail": "Higher moduli generalization"},
        ],
    }


@pytest.fixture
def sample_extractions():
    """Multiple paper extractions for merge testing."""
    return {
        "paper_a.pdf": {
            "concepts": [
                {"name": "Rogers-Ramanujan identities", "type": "identity",
                 "description": "Fundamental partition identities"},
                {"name": "Bailey lemma", "type": "theorem",
                 "description": "Framework for proving RR-type identities"},
            ],
            "relationships": [
                {"source": "Bailey lemma", "target": "Rogers-Ramanujan identities",
                 "relation": "proves", "detail": "Provides systematic proof"},
            ],
        },
        "paper_b.pdf": {
            "concepts": [
                {"name": "Rogers-Ramanujan identities", "type": "identity",
                 "description": "A pair of identities in the theory of partitions"},
                {"name": "Schur functions", "type": "object",
                 "description": "Symmetric functions in representation theory"},
                {"name": "Bailey's lemma", "type": "theorem",
                 "description": "Key lemma for partition identities"},
            ],
            "relationships": [
                {"source": "Bailey's lemma", "target": "Rogers-Ramanujan identities",
                 "relation": "proves", "detail": "Different proof approach"},
                {"source": "Schur functions", "target": "Rogers-Ramanujan identities",
                 "relation": "related_to", "detail": "Connection via symmetric functions"},
            ],
        },
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for embedding and chat requests."""
    client = MagicMock()

    # Mock embeddings
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 1536
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    client.embeddings.create.return_value = mock_response

    # Mock chat completions (concept extraction)
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "concepts": [
            {"name": "Rogers-Ramanujan identities", "type": "identity",
             "description": "Fundamental partition identities"},
        ],
        "relationships": [],
    })
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_chat_response

    return client


@pytest.fixture
def tmp_session_dir(tmp_path):
    """Create a temporary session directory structure."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "files").mkdir()
    return session_dir
