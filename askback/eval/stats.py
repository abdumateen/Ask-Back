"""§4.5 A14 statistics: H1/H2 on the quantile estimator T_p(v).

H1: at v (primary 0.10, fallback 0.05), T_P3(v) < T_P0/P1/P2(v) on S0,
     paired bootstrap on the quantile difference, Holm across the 3 baselines.
H2: on the policy in {P0, P2, P3} maximizing (T_S0 − T_S2(0.7)) at v,
     T_S2(0.7)(v) < T_S0(v); T_S2(1.0) is reported as a reference.
T_p(v) is the A14 quantile; a bootstrap resample whose group cannot reach v
contributes no evidence. If T_p(0.10) is undefined for some policy the test
moves to v = 0.05; if T_p(0.05) is also undefined, the run is vacuous and the
report must file A15 (H1 on a cooperative-persona cell).

Output: ``stats.json`` written beside the run's ``episodes.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from askback.runner.cli import _read_jsonl, load_config, stop_points
from askback.eval.frontier import replay_episodes, PRIMARY_V, FALLBACK_V

N_RESAMPLES = 10000
SEED = 20250
H1_BASELINES = ("P0", "P1", "P2")
H2_POLICIES = ("P0", "P2", "P3")
ALL_POLICIES = ("P0", "P1", "P2", "P3")


def quantile(turns, valid, v, total=None):
    """Smallest turn budget at which ≥ v fraction pass (A14). None if unattainable."""
    vt = sorted(t for t, ok in zip(turns, valid) if ok)
    n = total if total is not None else len(turns)
    need = math.ceil(v * n)
    return vt[need - 1] if len(vt) >= need else None


def gather(run_dir):
    """Per-policy, per-operating-point per-episode (episode_idx -> (turns, valid))."""
    run = Path(run_dir)
    cfg = load_config(run / "config.json")
    rows = _read_jsonl(run / "episodes.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
    points = stop_points(cfg)
    data = {}
    for policy, eps in groups.items():
        data[policy] = {}
        for rule, c, tau, param in points:
            key = f"{rule}:{param}"
            data[policy][key] = {
                idx: (t, v) for idx, t, v in replay_episodes(eps, rule, c, tau)
            }
    return data


def _envelope_at(data, common, v):
    turns = [data[i][0] for i in common]
    valid = [data[i][1] for i in common]
    return quantile(turns, valid, v, total=len(common))


def s0_T_defined(data, v):
    for pol in data:
        d = data[pol].get("S0:None")
        if not d:
            return False
        idx = list(d.keys())
        if quantile([d[i][0] for i in idx], [d[i][1] for i in idx], v) is None:
            return False
    return True


def choose_v(data):
    if s0_T_defined(data, PRIMARY_V):
        return PRIMARY_V, "primary 0.10"
    if s0_T_defined(data, FALLBACK_V):
        return FALLBACK_V, "fallback 0.05: T_p(0.10) undefined for some policy"
    return None, "vacuous: T_p(0.05) undefined for some policy; file A15"


def bootstrap_quantile_diff(Ta, Tb, common, v, n_resamples=N_RESAMPLES, rng=None):
    """One-sided p that Tb(v) < Ta(v) (b saves turns), paired by episode_idx.

    Resample indices (with replacement) from ``common``; per resample recompute
    T_a(v) and T_b(v); diff = T_a(v) − T_b(v). A resample where either group
    cannot reach v contributes no evidence. p = P(diff ≤ 0), +1 smoothing.
    """
    n = len(common)
    if rng is None:
        rng = np.random.default_rng(SEED)
    obs_a = _envelope_at(Ta, common, v)
    obs_b = _envelope_at(Tb, common, v)
    obs = (obs_a - obs_b) if (obs_a is not None and obs_b is not None) else None
    counts = 0
    valid_r = 0
    for _ in range(n_resamples):
        sel = [common[i] for i in rng.integers(0, n, size=n)]
        a = _envelope_at(Ta, sel, v)
        b = _envelope_at(Tb, sel, v)
        if a is None or b is None:
            continue
        valid_r += 1
        if a - b <= 0.0:
            counts += 1
    p = (counts + 1.0) / (valid_r + 1.0) if valid_r else 1.0
    return {"observed_quantile_diff": obs, "p_one_sided": p}


def holm_bonferroni(p_values):
    """Holm step-up correction; returns adjusted p-values in input order."""
    p = np.asarray(p_values, dtype=float)
    k = p.size
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (k - rank) * p[i])
        adj[i] = min(1.0, running)
    return [float(x) for x in adj]


def h1(data, v, rng):
    p3 = data.get("P3", {}).get("S0:None")
    if not p3:
        return None
    out = []
    for baseline in H1_BASELINES:
        b = data.get(baseline, {}).get("S0:None")
        if not b:
            continue
        common = sorted(set(p3) & set(b))
        if not common:
            continue
        boot = bootstrap_quantile_diff(b, p3, common, v, rng=rng)
        out.append({"baseline": baseline, "n_paired": len(common), **boot})
    if not out:
        return None
    out.sort(key=lambda c: c["p_one_sided"])
    raw = [c["p_one_sided"] for c in out]
    for c, p in zip(out, holm_bonferroni(raw)):
        c["p_holm"] = p
    return out


def h2(data, v, rng):
    s0_key, s2_key, s2_ref_key = "S0:None", "S2:0.7", "S2:1.0"
    savings = {}
    for pol in H2_POLICIES:
        a = data.get(pol, {}).get(s0_key)
        b = data.get(pol, {}).get(s2_key)
        if not a or not b:
            continue
        common = sorted(set(a) & set(b))
        if not common:
            continue
        sa = _envelope_at(a, common, v)
        sb = _envelope_at(b, common, v)
        if sa is None or sb is None:
            continue
        savings[pol] = (sa - sb, common)
    if not savings:
        return None
    pol, (save, common) = max(savings.items(), key=lambda kv: kv[1][0])
    boot = bootstrap_quantile_diff(
        data[pol][s0_key], data[pol][s2_key], common, v, rng=rng)
    ref = None
    s2_ref = data[pol].get(s2_ref_key)
    if s2_ref:
        rcommon = sorted(set(data[pol][s0_key]) & set(s2_ref))
        if rcommon:
            ref = {
                "T_S0": _envelope_at(data[pol][s0_key], rcommon, v),
                "T_S2_1_0": _envelope_at(s2_ref, rcommon, v),
            }
    return {
        "policy": pol,
        "turn_savings_S0_minus_S2": save,
        "n_paired": len(common),
        **boot,
        "reference_T_S2_1_0": ref,
    }


def stats(run_dir, n_resamples=N_RESAMPLES):
    data = gather(run_dir)
    rng = np.random.default_rng(SEED)
    v, reason = choose_v(data)
    policy_set = sorted(data.keys())
    full_set = set(ALL_POLICIES)
    result = {
        "method": "A14: paired bootstrap on quantile T_p(v), H1/H2",
        "n_resamples": n_resamples,
        "seed": SEED,
        "pairing": "episode_idx",
        "H1_v": v,
        "H2_v": v,
        "v_reason": reason,
    }
    if set(policy_set) == full_set:
        # A16b: full-policy-set run — behavior unchanged (byte-identity).
        result["vacuous"] = v is None
        if v is not None:
            result["H1"] = h1(data, v, rng)
            result["H2"] = h2(data, v, rng)
    else:
        # A16b addendum: H1/H2 computed only on full policy set. Subset runs
        # disclose the partial set and emit H1: [] / H2: null / vacuous: true.
        result["vacuous"] = True
        result["H1"] = []
        result["H2"] = None
        result["policy_set"] = policy_set
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m askback.eval.stats")
    ap.add_argument("run_dir", help="A1 output directory")
    args = ap.parse_args(argv)
    out = Path(args.run_dir) / "stats.json"
    out.write_text(json.dumps(stats(args.run_dir), sort_keys=True, indent=2)
                   + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
