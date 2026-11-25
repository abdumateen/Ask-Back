"""§14.2 `replay`: offline A2 replay over a finished A1 run directory.

Reads the config snapshot and episodes the `run` step wrote, drives every A2
operating point from the stored records, and writes `points.jsonl` beside them.
"""

from __future__ import annotations

import argparse
import sys

from askback.runner.cli import replay_study


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m askback.replay")
    ap.add_argument("--run", required=True, help="A1 output directory")
    args = ap.parse_args(argv)
    print(f"wrote {replay_study(args.run)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())