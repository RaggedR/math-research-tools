#!/usr/bin/env python3
"""
literature_agent.py — INSTINCT Literature Agent with progressive drill-down.

Searches Level 2 (meta-summaries) → Level 1 (concept summaries) → Level 0 (raw papers)
and synthesizes a compact answer with citations.

Usage:
    python3 literature_agent.py --dir <path> "What is the connection between X and Y?"
    python3 literature_agent.py --dir <path> --level 1 "What is the A2 Bailey lemma?"
    python3 literature_agent.py --dir <path> --level 0 "Exact statement of Theorem 3.2"
    python3 literature_agent.py --dir <path> --deep "crystal bases and q-series positivity"
"""

import sys
from pathlib import Path

from kg.config import load_config
from kg.agent import run_agent


def main():
    args = sys.argv[1:]

    if not args or '-h' in args or '--help' in args:
        print(__doc__)
        return

    # Parse options
    start_level = 2
    deep = False
    base_dir = None
    config_path = None
    question_parts = []

    i = 0
    while i < len(args):
        if args[i] == '--level' and i + 1 < len(args):
            start_level = int(args[i + 1])
            i += 2
        elif args[i] == '--deep':
            deep = True
            i += 1
        elif args[i] == '--dir' and i + 1 < len(args):
            base_dir = args[i + 1]
            i += 2
        elif args[i] == '--config' and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        else:
            question_parts.append(args[i])
            i += 1

    question = ' '.join(question_parts)
    if not question:
        print("Please provide a question.")
        return

    if base_dir is None:
        print("Error: --dir <path> is required")
        print(__doc__)
        sys.exit(1)

    cfg = load_config(config_path=config_path, data_dir=base_dir)

    print(f"\n{'=' * 60}")
    print(f"  LITERATURE AGENT QUERY")
    print(f"  Domain: {cfg.name}")
    print(f"  Q: {question}")
    print(f"{'=' * 60}")

    result = run_agent(
        question=question,
        base_dir=base_dir,
        config=cfg,
        start_level=start_level,
        deep=deep,
    )

    if result["answer"] is None:
        print("\n  No relevant passages found at any level.")
        return

    print(f"\n{'=' * 60}")
    print(f"  ANSWER")
    print(f"{'=' * 60}\n")
    print(result["answer"])
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
