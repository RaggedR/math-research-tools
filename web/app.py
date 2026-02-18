"""FastAPI web application for knowledge graph construction.

Provides file upload, WebSocket progress, and graph visualization endpoints.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from kg.config import SUPPORTED_EXTENSIONS
from kg.extract import extract_concepts, select_representative_chunks
from kg.graph import build_graph, merge_extractions, prepare_viz_data
from kg.ingest import extract_file, chunk_text, get_embeddings, ingest_files

logger = logging.getLogger(__name__)

# In-memory session store
sessions: dict[str, dict] = {}

# WebSocket connections per session
ws_connections: dict[str, list[WebSocket]] = {}

# Base directory for session files
SESSIONS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "sessions"

MAX_FILES = 80
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB per file
MAX_TOTAL_SIZE = 500 * 1024 * 1024  # 500 MB total per upload
SESSION_MAX_AGE_HOURS = 24
MAX_SESSIONS = 100


def cleanup_old_sessions():
    """Remove sessions older than SESSION_MAX_AGE_HOURS and enforce MAX_SESSIONS."""
    import shutil

    now = time.time()
    max_age_secs = SESSION_MAX_AGE_HOURS * 3600

    # Clean up in-memory sessions
    expired = [
        sid for sid, s in sessions.items()
        if now - s.get("created_at", now) > max_age_secs
    ]
    for sid in expired:
        sessions.pop(sid, None)
        ws_connections.pop(sid, None)

    # Enforce max session count (evict oldest first)
    if len(sessions) > MAX_SESSIONS:
        by_age = sorted(sessions.items(), key=lambda x: x[1].get("created_at", 0))
        for sid, _ in by_age[: len(sessions) - MAX_SESSIONS]:
            sessions.pop(sid, None)
            ws_connections.pop(sid, None)

    # Clean up on-disk session directories
    if SESSIONS_DIR.exists():
        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue
            # Remove if older than max age or if no matching in-memory session
            age = now - session_dir.stat().st_mtime
            if age > max_age_secs or session_dir.name not in sessions:
                try:
                    shutil.rmtree(session_dir)
                except OSError as e:
                    logger.warning("Failed to clean up %s: %s", session_dir, e)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Start periodic session cleanup on server startup."""
        async def _cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # Run every hour
                try:
                    cleanup_old_sessions()
                except Exception as e:
                    logger.warning("Cleanup error: %s", e)
        task = asyncio.create_task(_cleanup_loop())
        yield
        task.cancel()

    app = FastAPI(title="Knowledge Graph Builder", lifespan=lifespan)

    static_dir = Path(__file__).resolve().parent / "static"

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index_path = static_dir / "index.html"
        return HTMLResponse(content=index_path.read_text())

    @app.post("/api/upload")
    async def upload_files(files: list[UploadFile] = File(...)):
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > MAX_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files ({len(files)}). Maximum is {MAX_FILES}."
            )

        # Filter to supported file types
        valid_files = []
        for f in files:
            ext = Path(f.filename or "").suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                valid_files.append(f)

        if not valid_files:
            raise HTTPException(
                status_code=400,
                detail="No supported files found. Accepted types: PDF, TXT, MD"
            )

        # Create session
        session_id = str(uuid.uuid4())
        session_dir = SESSIONS_DIR / session_id
        files_dir = session_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        # Save files with size limits and path traversal protection
        saved_paths = []
        total_size = 0
        for f in valid_files:
            # Sanitize filename: strip directory components to prevent path traversal
            safe_name = Path(f.filename or "unknown").name
            if not safe_name or safe_name.startswith("."):
                continue

            # Read with size limit
            content = await f.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{safe_name}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"
                )
            total_size += len(content)
            if total_size > MAX_TOTAL_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Total upload exceeds {MAX_TOTAL_SIZE // (1024*1024)}MB limit"
                )

            file_path = files_dir / safe_name
            file_path.write_bytes(content)
            saved_paths.append(file_path)

        # Register session
        sessions[session_id] = {
            "session_id": session_id,
            "status": "processing",
            "file_count": len(saved_paths),
            "files": [p.name for p in saved_paths],
            "session_dir": str(session_dir),
            "graph": None,
            "error": None,
            "created_at": time.time(),
        }

        # Start background processing
        process_session_background(session_id, saved_paths, session_dir)

        return {
            "session_id": session_id,
            "status": "processing",
            "file_count": len(saved_paths),
        }

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = sessions[session_id]
        return {
            "session_id": session["session_id"],
            "status": session["status"],
            "file_count": session["file_count"],
            "files": session["files"],
            "error": session["error"],
        }

    @app.get("/api/graph/{session_id}")
    async def get_graph(session_id: str):
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = sessions[session_id]
        if session["status"] == "error":
            raise HTTPException(status_code=500, detail=session["error"])

        if session["status"] != "complete" or session["graph"] is None:
            return JSONResponse(
                status_code=202,
                content={"status": "processing", "message": "Graph not ready yet"}
            )

        return session["graph"]

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        await websocket.accept()

        if session_id not in ws_connections:
            ws_connections[session_id] = []
        ws_connections[session_id].append(websocket)

        try:
            # If already complete, send immediately
            if session_id in sessions:
                session = sessions[session_id]
                if session["status"] == "complete":
                    await websocket.send_json({
                        "type": "complete",
                        "graph_url": f"/api/graph/{session_id}",
                    })
                elif session["status"] == "error":
                    await websocket.send_json({
                        "type": "error",
                        "message": session["error"],
                    })

            # Keep connection open until client disconnects
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if session_id in ws_connections:
                ws_connections[session_id] = [
                    ws for ws in ws_connections[session_id] if ws != websocket
                ]

    # Mount static files last so API routes take precedence
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def process_session_background(session_id: str, file_paths: list[Path],
                                session_dir: Path):
    """Start background processing for a session.

    In production, this spawns a background task. For testing, it can be mocked.
    """
    loop = asyncio.get_running_loop()
    loop.create_task(_process_session(session_id, file_paths, session_dir))


async def _broadcast_progress(session_id: str, stage: str, detail: str, percent: float):
    """Send progress update to all WebSocket clients for a session."""
    if session_id not in ws_connections:
        return

    message = {
        "type": "progress",
        "stage": stage,
        "detail": detail,
        "percent": round(percent, 1),
    }

    dead = []
    for ws in ws_connections.get(session_id, []):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)

    # Clean up dead connections
    if dead and session_id in ws_connections:
        ws_connections[session_id] = [
            ws for ws in ws_connections[session_id] if ws not in dead
        ]


async def _process_session(session_id: str, file_paths: list[Path],
                            session_dir: Path):
    """Process uploaded files into a knowledge graph.

    Runs the full pipeline: extract text -> chunk -> embed -> extract concepts
    -> merge -> build graph. Sends progress updates via WebSocket.
    """
    try:
        from openai import OpenAI
        openai_client = OpenAI()

        total_files = len(file_paths)

        # Stage 1: Ingest — extract text and chunk
        await _broadcast_progress(session_id, "ingesting", "Starting...", 0)

        all_chunks_by_file = {}
        for idx, file_path in enumerate(file_paths):
            pct = (idx / total_files) * 33
            await _broadcast_progress(
                session_id, "ingesting",
                f"{file_path.name} ({idx + 1}/{total_files})", pct
            )

            pages = await asyncio.to_thread(extract_file, file_path)
            if pages:
                chunks = chunk_text(pages)
                if chunks:
                    all_chunks_by_file[file_path.name] = chunks

        await _broadcast_progress(session_id, "ingesting", "Complete", 33)

        if not all_chunks_by_file:
            sessions[session_id]["status"] = "error"
            sessions[session_id]["error"] = "No text could be extracted from uploaded files"
            await _broadcast_ws_error(session_id, sessions[session_id]["error"])
            return

        # Stage 2: Extract concepts from each file
        await _broadcast_progress(session_id, "extracting", "Starting...", 33)

        all_extractions = {}
        file_names = sorted(all_chunks_by_file.keys())
        for idx, file_name in enumerate(file_names):
            chunks = all_chunks_by_file[file_name]
            pct = 33 + (idx / len(file_names)) * 34
            await _broadcast_progress(
                session_id, "extracting",
                f"{file_name} ({idx + 1}/{len(file_names)})", pct
            )

            selected = select_representative_chunks(chunks)
            text = "\n\n---\n\n".join(c["text"][:3000] for c in selected)
            if len(text) > 6000:
                text = text[:6000] + "\n[...truncated...]"

            extraction = await asyncio.to_thread(
                extract_concepts, text, file_name, openai_client
            )
            all_extractions[file_name] = extraction

        await _broadcast_progress(session_id, "extracting", "Complete", 67)

        # Stage 3: Build graph
        await _broadcast_progress(session_id, "building", "Merging extractions...", 67)

        concepts, edges = merge_extractions(all_extractions)
        graph = build_graph(concepts, edges)
        viz_data = prepare_viz_data(graph)

        await _broadcast_progress(session_id, "building", "Complete", 100)

        # Store result
        sessions[session_id]["status"] = "complete"
        sessions[session_id]["graph"] = viz_data

        # Save graph JSON to session dir
        graph_path = session_dir / "knowledge_graph.json"
        graph_path.write_text(json.dumps(graph, indent=2))

        # Notify WebSocket clients
        for ws in ws_connections.get(session_id, []):
            try:
                await ws.send_json({
                    "type": "complete",
                    "graph_url": f"/api/graph/{session_id}",
                })
            except Exception:
                pass

    except Exception as e:
        logger.exception("Error processing session %s", session_id)
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)
        await _broadcast_ws_error(session_id, str(e))


async def _broadcast_ws_error(session_id: str, message: str):
    """Send error to all WebSocket clients."""
    for ws in ws_connections.get(session_id, []):
        try:
            await ws.send_json({"type": "error", "message": message})
        except Exception:
            pass


# Create the app instance for uvicorn
app = create_app()
