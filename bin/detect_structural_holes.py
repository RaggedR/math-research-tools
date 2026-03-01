#!/usr/bin/env python3
"""
detect_structural_holes.py — Detect citation gaps between thematic clusters.

Usage:
    python3 detect_structural_holes.py --dir ~/data/arxiv-rag/
    python3 detect_structural_holes.py --dir ~/data/arxiv-rag/ --skip-fetch
    python3 detect_structural_holes.py --dir ~/data/arxiv-rag/ --top-k 5
    python3 detect_structural_holes.py --dir ~/data/arxiv-rag/ --config configs/math.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from kg.config import load_config
from kg.structural_holes import run_structural_holes

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Detect structural holes in a research paper corpus")
    parser.add_argument("--dir", required=True, type=Path,
                        help="Base directory containing paper PDFs and knowledge_graph.json")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip Semantic Scholar fetching, use cached citations only")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Number of top structural holes to analyze (default: 15)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to domain YAML config file")
    args = parser.parse_args()

    base_dir = args.dir.expanduser().resolve()
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    config = load_config(config_path=args.config, data_dir=base_dir)
    print(f"Domain: {config.name}")

    run_structural_holes(
        base_dir=base_dir,
        config=config,
        skip_fetch=args.skip_fetch,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
