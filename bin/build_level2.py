#!/usr/bin/env python3
"""
build_level2.py — Build Level 2 of the INSTINCT literature hierarchy.

1. Index all Level 1 summaries into ChromaDB
2. Identify research themes via LLM analysis
3. Generate meta-summaries for each theme
4. Build Level 2 knowledge graph with interactive visualization

Usage:
    python3 build_level2.py --dir papers              # Full pipeline
    python3 build_level2.py --dir papers --index-only  # Just index
    python3 build_level2.py --dir papers --themes-only # Just themes
    python3 build_level2.py --dir papers --meta-only   # Just meta-summaries
    python3 build_level2.py --dir papers --config configs/evo.yaml
"""

import sys
from pathlib import Path

from kg.config import load_config
from kg.level2 import run_level2


def main():
    args = sys.argv[1:]

    base_dir = Path("papers")
    config_path = None

    for i, a in enumerate(args):
        if a == '--dir' and i + 1 < len(args):
            base_dir = Path(args[i + 1])
        elif a == '--config' and i + 1 < len(args):
            config_path = args[i + 1]

    arg_set = set(args)
    config = load_config(config_path=config_path, data_dir=base_dir)
    print(f"Domain: {config.name}")

    run_level2(
        base_dir=base_dir,
        config=config,
        index_only="--index-only" in arg_set,
        themes_only="--themes-only" in arg_set,
        meta_only="--meta-only" in arg_set,
    )


if __name__ == "__main__":
    main()
