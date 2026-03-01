#!/usr/bin/env python3
"""
generate_summaries.py — Generate Level 1 concept summaries.

For each concept in the knowledge graph (degree >= 2), queries ChromaDB
for relevant passages and uses an LLM to write a 2-3 page summary.

Usage:
    python3 generate_summaries.py --dir papers --all
    python3 generate_summaries.py --dir papers --start 0 --end 40
    python3 generate_summaries.py --dir papers --all --model gpt-4o
    python3 generate_summaries.py --dir papers --all --config configs/evo.yaml
"""

import sys
from pathlib import Path

from kg.config import load_config
from kg.summaries import run_summaries, DEFAULT_MODEL


def main():
    args = sys.argv[1:]

    start_idx = 0
    end_idx = None
    model = DEFAULT_MODEL
    base_dir = Path("papers")
    config_path = None

    i = 0
    while i < len(args):
        if args[i] == '--start':
            start_idx = int(args[i + 1])
            i += 2
        elif args[i] == '--end':
            end_idx = int(args[i + 1])
            i += 2
        elif args[i] == '--all':
            start_idx = 0
            end_idx = None
            i += 1
        elif args[i] == '--model':
            model = args[i + 1]
            i += 2
        elif args[i] == '--dir':
            base_dir = Path(args[i + 1])
            i += 2
        elif args[i] == '--config':
            config_path = args[i + 1]
            i += 2
        else:
            i += 1

    config = load_config(config_path=config_path, data_dir=base_dir)
    print(f"Domain: {config.name}")

    run_summaries(
        base_dir=base_dir,
        config=config,
        start_idx=start_idx,
        end_idx=end_idx,
        model=model,
    )


if __name__ == '__main__':
    main()
