# INSTINCT — Integrated Synthesis & Taxonomy for Intelligent Navigation of Conceptual Terrain

A unified Python toolkit for research literature analysis: ingest papers, extract concepts, build knowledge graphs, generate summaries, detect structural holes, and produce LaTeX survey papers.

[Live demo: knowledge graph built from q-series and Rogers-Ramanujan identity papers](https://raggedr.github.io/math-research-tools/knowledge_graph.html)

[Web app: upload PDFs and build knowledge graphs interactively](https://kg-web-314185672280.australia-southeast1.run.app/)

## Pipeline

```
lit_review.py → build_knowledge_graph.py → generate_summaries.py → build_level2.py → detect_structural_holes.py → generate_survey.py
   (arXiv)        (L0: concepts/edges)       (L1: summaries)       (L2: themes)        (citation gaps)           (LaTeX paper)
```

Each stage reads from and writes to a single data directory. The `literature_agent.py` provides progressive drill-down search across all levels.

## Setup

```bash
pip install -e .
```

You need API keys in your environment:
- `OPENAI_API_KEY` — required for all stages
- `ANTHROPIC_API_KEY` — required for survey generation

## Domain Configuration

INSTINCT is domain-agnostic. Each data directory can have a `domain.yaml` that points to a config:

```yaml
# ~/data/genetic/domain.yaml
config: evolutionary-computation
```

Three-tier config resolution:
1. Explicit `--config path.yaml` flag (highest priority)
2. `domain.yaml` in the data directory
3. Fallback to `configs/math.yaml`

Built-in configs: `math`, `evolutionary-computation`, `knowledge-graphs`.

## CLI Tools

| Command | Purpose |
|---------|---------|
| `bin/lit_review.py` | Search arXiv, rank papers, download PDFs |
| `bin/build_knowledge_graph.py` | Extract concepts + relationships, build D3.js visualization |
| `bin/generate_summaries.py` | Generate Level 1 concept summaries via LLM |
| `bin/build_level2.py` | Identify themes, generate meta-summaries, build L2 graph |
| `bin/detect_structural_holes.py` | Citation gap detection between thematic clusters |
| `bin/generate_survey.py` | Generate a complete LaTeX survey paper |
| `bin/literature_agent.py` | Progressive drill-down search (L2 → L1 → L0) |

All commands accept `--dir <path>` and `--config <path>`.

## Architecture

```
kg/                     # Core library
  config.py             # DomainConfig dataclass + YAML loader
  llm.py                # OpenAI + Anthropic adapters with retry
  utils.py              # Shared utilities (slugify)
  ingest.py             # PDF/text ingestion + ChromaDB storage
  extract.py            # GPT-4o-mini concept extraction
  graph.py              # Merge, deduplicate, build graph
  visualize.py          # D3.js HTML visualization
  summaries.py          # Level 1 concept summaries
  level2.py             # Level 2 themes + meta-summaries
  structural_holes.py   # Citation gap detection
  survey.py             # LaTeX survey generation
  agent.py              # Literature search agent

configs/                # Domain YAML configurations
  math.yaml
  evolutionary-computation.yaml
  knowledge-graphs.yaml

bin/                    # Thin CLI wrappers
web/                    # FastAPI web app (upload + visualize)
tests/                  # Test suite
```

## Testing

```bash
pytest tests/ -v
```

## Web App

```bash
uvicorn web.app:app --reload
```

Upload PDFs and build knowledge graphs interactively at http://127.0.0.1:8000.

## Claude Code Integration

Copy `commands/` files to `~/.claude/commands/` for `/lit-review` and `/knowledge-graph` slash commands.
