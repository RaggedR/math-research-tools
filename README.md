# math-research-tools

Claude Code skills and Python scripts for mathematical literature review and knowledge graph construction.

## What's included

| Component | Purpose |
|-----------|---------|
| `kg/` | Library package: PDF/text ingestion, GPT-4o-mini concept extraction, graph building |
| `web/` | FastAPI web app: file upload, WebSocket progress, D3.js visualization |
| `bin/lit_review.py` | Search arXiv, rank papers by embedding similarity, download PDFs |
| `bin/build_knowledge_graph.py` | CLI wrapper for the knowledge graph pipeline |
| `commands/lit-review.md` | Claude Code skill for multi-phase literature review |
| `commands/knowledge-graph.md` | Claude Code skill for knowledge graph construction |

## Setup

```bash
pip install -r requirements.txt
```

For development (includes test dependencies):

```bash
pip install -r requirements-dev.txt
playwright install chromium  # for E2E tests
```

You need an OpenAI API key in your environment (`OPENAI_API_KEY`).

## Usage

### Web app (recommended)

Start the web server and upload files through the browser:

```bash
cd /path/to/math-research-tools
uvicorn web.app:app --reload
```

Then open http://127.0.0.1:8000 in your browser. Upload PDF, TXT, or MD files (up to 80), and watch the knowledge graph build in real time via WebSocket progress updates.

### CLI

Build a knowledge graph from a directory of PDFs:

```bash
python3 bin/build_knowledge_graph.py --dir /tmp/my-review
```

If the directory has `papers/*.pdf` but no `chroma_db/`, the script automatically ingests the PDFs first. Use `--resume` to continue an interrupted build, or `--viz-only` to regenerate the HTML visualization.

### Literature review

Search arXiv for papers, rank by relevance, download PDFs:

```bash
python3 bin/lit_review.py search "cylindric partitions positivity" --dir /tmp/my-review --max-papers 20
```

Run multiple searches to the same directory — results accumulate (deduplication by arXiv ID, highest similarity score kept).

After collecting papers from multiple queries, clean up to keep only the top N:

```bash
python3 bin/lit_review.py cleanup --dir /tmp/my-review --keep 50 "cylindric partitions q-series positivity"
```

List existing reviews in a directory:

```bash
python3 bin/lit_review.py list --dir /tmp
```

## Architecture

```
kg/             # Library package
  config.py     # Constants, prompts, normalization tables
  ingest.py     # PDF/text extraction, chunking, embedding
  extract.py    # GPT-4o-mini concept extraction
  graph.py      # Merge, deduplicate, build graph + viz data
  visualize.py  # Standalone HTML generation (CLI)

web/            # FastAPI web app
  app.py        # Endpoints: upload, sessions, graph, WebSocket
  static/       # Frontend: index.html, app.js, style.css

tests/          # Test suite (pytest)
  test_ingest.py, test_extract.py, test_graph.py  # Unit tests
  test_api.py                                      # API tests
  test_e2e.py                                      # Playwright E2E tests
```

## Testing

```bash
# Unit + API tests
pytest tests/ -v

# E2E tests (requires Playwright)
pytest tests/test_e2e.py -v
```

## Claude Code integration

Copy the `commands/` files into your Claude Code commands directory (`~/.claude/commands/`) to enable the `/lit-review` and `/knowledge-graph` slash commands. Update the `<repo>` placeholder in the skill files to point to where you cloned this repository.

### Workflow

1. `/lit-review /tmp/my-review topic keywords` — Claude runs a multi-phase search: conversation mining, diversified arXiv queries, abstract snowball, Semantic Scholar citation chasing, and cleanup
2. `/knowledge-graph /tmp/my-review` — Ingest PDFs, extract concepts, build interactive visualization
