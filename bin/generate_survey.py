#!/usr/bin/env python3
"""
generate_survey.py — Generate a survey paper from the INSTINCT hierarchy.

Usage:
    python3 generate_survey.py --dir <path>                      # Markdown (default)
    python3 generate_survey.py --dir <path> --format latex       # LaTeX + .bib
    python3 generate_survey.py --dir <path> --force              # Ignore cache
    python3 generate_survey.py --dir <path> --outline-only       # Just outline
    python3 generate_survey.py --dir <path> --section sec-id     # One section
    python3 generate_survey.py --dir <path> --config evo.yaml    # Custom domain
"""

import argparse
import sys
from pathlib import Path

from kg.config import load_config
from kg.survey import run_survey, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(
        description="Generate a survey paper from the INSTINCT hierarchy."
    )
    parser.add_argument("--dir", required=True, help="Base data directory")
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown",
                        help="Output format: markdown (default) or latex")
    parser.add_argument("--force", action="store_true", help="Ignore cache, regenerate all")
    parser.add_argument("--outline-only", action="store_true", help="Only generate the outline")
    parser.add_argument("--section", type=str, help="Regenerate a single section by ID")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--config", type=str, help="Path to domain config YAML")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, data_dir=args.dir)
    print(f"Domain: {cfg.name}")

    run_survey(
        base_dir=args.dir,
        config=cfg,
        force=args.force,
        outline_only=args.outline_only,
        section=args.section,
        model=args.model,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
