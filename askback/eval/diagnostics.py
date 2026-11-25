"""§4.7 A4 diagnostics on an A1 ``episodes.jsonl``.

- blind-spot targeting rate: fraction of asked fields that lie in the
  episode's *initial* gap set G0 (the fields that are unresolvable from the
  extraction alone, so a question there cannot resolve a rule). Computed
  purely from the log: ``records[].field`` vs ``initial.g``. Higher = worse.
- hit-validity Spearman: per-episode rank correlation between the resolvable
  ask share (1 − blind-spot share) and the episode's validity. NaN when the
  share or validity is constant across episodes.

Output: ``diagnostics.json`` written beside the run's ``episodes.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from askback.runner.cli import _read_jsonl

def episode_blind_share(row):
    """Share of asked fields that were in the initial gap set G0."""
    g0 = set(row["initial"]["g"])
    asked = [r["field"] for r in row["records"]]
    if not asked:
        return None, 0.0, 0
    blind = sum(1 for f in asked if f in g0)
    return blind / len(asked), blind / len(asked), len(asked)


def diagnostics(run_dir):
    run = Path(run_dir)
    rows = _read_jsonl(run / "episodes.jsonl")

    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)

    per_policy = {}
    overall_blind = 0.0
    overall_asked = 0
    for policy in sorted(groups):
        eps = groups[policy]
        blind_total = asked_total = 0
        share_x, validity_x = [], []
        for row in eps:
            share, blind_n, asked_n = episode_blind_share(row)
            asked_total += asked_n
            blind_total += round(blind_n * asked_n)
            if share is not None:
                share_x.append(share)
                validity_x.append(int(row["validity"]))
        rate = blind_total / asked_total if asked_total else None
        overall_blind += blind_total
        overall_asked += asked_total

        corr = None
        if len(share_x) >= 3 and len(set(validity_x)) > 1 and len(set(share_x)) > 1:
            corr = float(pd.Series(share_x).corr(pd.Series(validity_x), method="spearman"))

        per_policy[policy] = {
            "n_episodes": len(eps),
            "blind_spot_targeting_rate": rate,
            "total_asks": asked_total,
            "hit_validity_spearman": corr,
        }

    return {
        "method": "A4 diagnostics on log fields + initial gap set",
        "overall_blind_spot_targeting_rate": (
            overall_blind / overall_asked if overall_asked else None),
        "overall_total_asks": overall_asked,
        "policies": per_policy,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m askback.eval.diagnostics")
    ap.add_argument("run_dir", help="A1 output directory")
    args = ap.parse_args(argv)
    out = Path(args.run_dir) / "diagnostics.json"
    out.write_text(json.dumps(diagnostics(args.run_dir), sort_keys=True, indent=2)
                   + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
