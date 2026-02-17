---
allowed-tools: Bash
description: Build a knowledge graph + interactive visualization from a paper directory
user-invocable: true
---

# Knowledge Graph: $ARGUMENTS

Build a knowledge graph and interactive D3.js visualization from a directory of PDFs.

## Instructions

1. Determine the target directory (required):
   - If `$ARGUMENTS` is a path, use it directly
   - If `$ARGUMENTS` is empty, tell the user: "Please provide a directory. Example: `/knowledge-graph /tmp/my-review`"

2. Run:

```bash
python3 <repo>/bin/build_knowledge_graph.py --dir <target-directory>
```
where `<repo>` is the path to this repository (e.g., `~/git/math-research-tools`).

   Set timeout to 600000 (10 minutes) — GPT-4o-mini extraction can take a while for many papers.

   **Auto-ingestion**: If the directory has `papers/*.pdf` but no `chroma_db/`, the script automatically ingests the PDFs into ChromaDB first, then builds the knowledge graph. No separate step needed.

   Use `--resume` to continue an interrupted build (reuses cached extractions).
   Use `--viz-only` to regenerate the HTML from an existing `knowledge_graph.json`.

3. Report results:
   - Number of concepts and edges
   - Location of the HTML file
   - Remind user: click nodes to highlight, search box to find concepts, double-click to reset

## Notes
- The `--dir` flag is **required** — the script has no default directory
- Requires: `pip install PyMuPDF chromadb openai` (same deps as lit-review)
