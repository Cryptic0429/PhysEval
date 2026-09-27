#!/usr/bin/env python3
"""PhysEval entry point. Optional comparison code is loaded only on request."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "eval": "scripts/run_batch_simple.py",
    "score": "scripts/score_results.py",
    "compare": "compare/scripts/run_compare.py",
    "compare-summary": "compare/scripts/summarize_compare.py",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PhysEval: physical evaluation and optional tracking comparison.",
        epilog="Use <command> --help for command-specific options. Compare requires requirements-compare.txt.",
    )
    parser.add_argument("command", choices=COMMANDS,
                        help="eval / score: core pipeline; compare / compare-summary: optional comparison")
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    script = PROJECT_ROOT / COMMANDS[args.command]
    previous_argv = sys.argv
    try:
        sys.argv = [str(script), *args.args]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
