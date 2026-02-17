# math-research-tools

Claude Code skills and Python scripts for mathematical literature review and knowledge graph construction.

## What's included

| File | Purpose |
|------|---------|
| `bin/lit_review.py` | Search arXiv, rank papers by embedding similarity, download PDFs |
| `bin/build_knowledge_graph.py` | Ingest PDFs into ChromaDB, extract concepts via GPT-4o-mini, build interactive D3.js knowledge graph |
| `commands/lit-review.md` | Claude Code skill for multi-phase literature review |
| `commands/knowledge-graph.md` | Claude Code skill for knowledge graph construction |

## Setup

```bash
pip install -r requirements.txt
```

You need an OpenAI API key in your environment (`OPENAI_API_KEY`).

## Usage

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

### Knowledge graph

Build an interactive knowledge graph from a directory of PDFs:

```bash
python3 bin/build_knowledge_graph.py --dir /tmp/my-review
```

If the directory has `papers/*.pdf` but no `chroma_db/`, the script automatically ingests the PDFs first. Use `--resume` to continue an interrupted build, or `--viz-only` to regenerate the HTML visualization.

## Claude Code integration

Copy the `commands/` files into your Claude Code commands directory (`~/.claude/commands/`) to enable the `/lit-review` and `/knowledge-graph` slash commands. Update the `<repo>` placeholder in the skill files to point to where you cloned this repository.

### Workflow

1. `/lit-review /tmp/my-review topic keywords` — Claude runs a multi-phase search: conversation mining, diversified arXiv queries, abstract snowball, Semantic Scholar citation chasing, and cleanup
2. `/knowledge-graph /tmp/my-review` — Ingest PDFs, extract concepts, build interactive visualization
