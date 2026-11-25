"""§4.3 per-policy bounds metrics from an A1 ``episodes.jsonl``.

Every episode row carries ``bound_oracle_questions`` (oracle minimum questions,
``None`` when the cell is unpassable) and ``bound_no_interaction_pass``
(turn-0 validity), persisted by :func:`askback.runner.cli._episode_row` via
:func:`askback.runner.oracle.episode_bounds` (PRD §12, directive step 3).

For each policy:

  n_episodes             rows for this policy
  n_passable             rows where ``bound_oracle_questions`` is not None
  n_episodes_unpassable  rows where it is None
  mean_bound_oracle_questions mean over passable rows, ``None`` when none pass

A cell is *passable* exactly when the oracle finds some subset of the knowledge
set whose true values make the tier-k validator pass; ``None`` means unpassable
even with the oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_episodes(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def policy_metrics(episodes: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for ep in episodes:
        groups.setdefault(ep["policy"], []).append(ep)

    report: dict[str, dict] = {}
    for policy in sorted(groups):
        rows = groups[policy]
        oracle = [
            ep["bound_oracle_questions"]
            for ep in rows
            if ep["bound_oracle_questions"] is not None
        ]
        report[policy] = {
            "n_episodes": len(rows),
            "n_passable": len(oracle),
            "n_episodes_unpassable": len(rows) - len(oracle),
            "mean_bound_oracle_questions": (
                sum(oracle) / len(oracle) if oracle else None
            ),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m askback.metrics")
    ap.add_argument("path", help="episodes.jsonl from an askback.run output dir")
    args = ap.parse_args(argv)
    print(json.dumps(policy_metrics(load_episodes(args.path)),
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
