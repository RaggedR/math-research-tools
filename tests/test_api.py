"""Tests for web.app — FastAPI endpoints and WebSocket."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def app():
    """Create a fresh FastAPI app for each test."""
    from web.app import create_app

    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_root_serves_html(client):
    """GET / should serve the index.html page."""
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Knowledge Graph" in response.text


@pytest.mark.asyncio
async def test_upload_creates_session(client):
    """POST /api/upload with files should create a session."""
    files = [
        ("files", ("sample.txt", open(FIXTURES_DIR / "sample.txt", "rb"), "text/plain")),
    ]
    with patch("web.app.process_session_background") as mock_process:
        response = await client.post("/api/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "processing"
    assert data["file_count"] == 1


@pytest.mark.asyncio
async def test_upload_rejects_no_files(client):
    """POST /api/upload with no files should return 400."""
    response = await client.post("/api/upload")
    assert response.status_code == 422 or response.status_code == 400


@pytest.mark.asyncio
async def test_upload_rejects_too_many_files(client):
    """POST /api/upload with > 80 files should return 400."""
    files = [
        ("files", (f"file{i}.txt", b"content", "text/plain"))
        for i in range(81)
    ]
    response = await client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "too many" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_types(client):
    """POST /api/upload with only unsupported file types should return 400."""
    files = [
        ("files", ("test.exe", b"binary", "application/octet-stream")),
    ]
    response = await client.post("/api/upload", files=files)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_session_status_not_found(client):
    """GET /api/sessions/{id} for nonexistent session returns 404."""
    response = await client.get("/api/sessions/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_session_status_after_upload(client):
    """GET /api/sessions/{id} returns status for valid session."""
    files = [
        ("files", ("sample.txt", open(FIXTURES_DIR / "sample.txt", "rb"), "text/plain")),
    ]
    with patch("web.app.process_session_background") as mock_process:
        upload_resp = await client.post("/api/upload", files=files)

    session_id = upload_resp.json()["session_id"]
    response = await client.get(f"/api/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert "status" in data
    assert "file_count" in data


@pytest.mark.asyncio
async def test_graph_not_found(client):
    """GET /api/graph/{id} for nonexistent session returns 404."""
    response = await client.get("/api/graph/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_graph_not_ready(client):
    """GET /api/graph/{id} before processing completes returns 202."""
    files = [
        ("files", ("sample.txt", open(FIXTURES_DIR / "sample.txt", "rb"), "text/plain")),
    ]
    with patch("web.app.process_session_background") as mock_process:
        upload_resp = await client.post("/api/upload", files=files)

    session_id = upload_resp.json()["session_id"]
    response = await client.get(f"/api/graph/{session_id}")
    # Not yet complete
    assert response.status_code == 202


@pytest.mark.asyncio
async def test_full_pipeline_with_mock(client, app):
    """Upload files, simulate processing, fetch graph."""
    from web.app import sessions

    files = [
        ("files", ("sample.txt", open(FIXTURES_DIR / "sample.txt", "rb"), "text/plain")),
    ]
    with patch("web.app.process_session_background") as mock_process:
        upload_resp = await client.post("/api/upload", files=files)

    session_id = upload_resp.json()["session_id"]

    # Simulate completed processing by setting graph data
    sessions[session_id]["status"] = "complete"
    sessions[session_id]["graph"] = {
        "nodes": [
            {"id": "test", "label": "Test", "type": "object",
             "papers": 1, "degree": 0, "description": "", "color": "#4A90D9"}
        ],
        "links": [],
    }

    response = await client.get(f"/api/graph/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert len(data["nodes"]) == 1
