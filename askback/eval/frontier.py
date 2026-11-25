"""§4.4 turns-validity envelope T_p(v) (PRD R3 / A14) from an A1 run directory.

For every A2 operating point (policy × stop rule × parameter) each episode is
replayed from its stored records, and T_p(v) is the smallest turn budget at
which at least v of the episodes pass. The v-grid is fixed at six points (A14);
the primary endpoint is T_p(0.10), with fallback T_p(0.05). A point whose best
reachable pass rate falls below v reports ``None`` there.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from askback.runner.cli import _read_jsonl, load_config, stop_points
from askback.runner.episode import InitialState, TurnRecord
from askback.stops.replay import replay

GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
PRIMARY_V = 0.10
FALLBACK_V = 0.05


def point_envelope(turns, valid, total, grid=GRID):
    """``turns``: per-episode turns after replay; ``valid``: parallel bools."""
    vt = sorted(t for t, ok in zip(turns, valid) if ok)
    out = {}
    for v in grid:
        need = math.ceil(v * total)
        out[str(v)] = vt[need - 1] if len(vt) >= need else None
    return out


def replay_episodes(eps, rule, c, tau):
    """Replay every episode row in ``eps`` under one stop point.

    Returns a list of ``(episode_idx, turns, valid)`` tuples so callers can
    pair resamples by episode index (A14 paired bootstrap).
    """
    out = []
    for row in eps:
        init = InitialState(
            u=frozenset(row["initial"]["u"]),
            g=frozenset(row["initial"]["g"]),
            delta_p=row["initial"]["delta_p"],
            validator_pass=row["initial"]["validator_pass"],
        )
        recs = [TurnRecord(**r) for r in row["records"]]
        o = replay(recs, rule=rule, c=c, tau=tau,
                   patience=row["patience"], initial=init)
        out.append((row["episode_idx"], o.turns, bool(o.validity)))
    return out


def frontier(run_dir, grid=GRID):
    run = Path(run_dir)
    cfg = load_config(run / "config.json")
    rows = _read_jsonl(run / "episodes.jsonl")
    points = stop_points(cfg)

    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)

    env = {}
    for policy, eps in groups.items():
        env[policy] = {}
        for rule, c, tau, param in points:
            key = f"{rule}:{param}"
            turns, valid = [], []
            for _, t, v in replay_episodes(eps, rule, c, tau):
                turns.append(t)
                valid.append(v)
            env[policy][key] = point_envelope(turns, valid, len(eps), grid)

    return {
        "grid": list(grid),
        "primary_v": PRIMARY_V,
        "fallback_v": FALLBACK_V,
        "policies": env,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m askback.eval.frontier")
    ap.add_argument("run_dir", help="A1 output directory")
    args = ap.parse_args(argv)
    print(json.dumps(frontier(args.run_dir), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
