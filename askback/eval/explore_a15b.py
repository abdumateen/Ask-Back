"""A15b exploratory: H1 at v=0.10 on the locked core_v1 cell, excluding P1.

The pre-registered H1 fell back to v=0.05 because P1's S0 T(0.10) was
undefined, which flattened the frontier. A15b tests P3 against the surviving
baselines {P0, P2} at the primary v = 0.10, paired bootstrap, one-sided,
Holm across the two baselines. EXPLICITLY EXPLORATORY / POST-HOC: not
confirmatory under the pre-registered realistic cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from askback.eval.stats import (
    SEED,
    bootstrap_quantile_diff,
    gather,
    holm_bonferroni,
)

V = 0.10
BASELINES = ("P0", "P2")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m askback.eval.explore_a15b")
    ap.add_argument("run_dir", help="A1 output directory with episodes.jsonl")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args(argv)

    data = gather(args.run_dir)
    p3 = data["P3"]["S0:None"]
    rng = np.random.default_rng(SEED)

    rows = []
    for baseline in BASELINES:
        base = data[baseline]["S0:None"]
        common = sorted(set(p3) & set(base))
        boot = bootstrap_quantile_diff(base, p3, common, V, rng=rng)
        rows.append({"baseline": baseline, "n_paired": len(common), **boot})

    raw = [r["p_one_sided"] for r in rows]
    for row, p in zip(rows, holm_bonferroni(raw)):
        row["p_holm"] = p

    result = {
        "method": "A15b: exploratory H1 at v=0.10 on realistic excl. P1; "
                  "P3 vs {P0,P2}; paired bootstrap; Holm alpha=0.05. "
                  "NOT confirmatory.",
        "v": V,
        "seed": SEED,
        "exploratory": True,
        "H1": rows,
    }
    path = Path(args.out or (Path(args.run_dir).parent / "exploratory" / "a15b.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                     encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
