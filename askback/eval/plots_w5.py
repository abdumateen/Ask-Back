"""W5 plots P1-P5 (matplotlib enabled in W5).

Reads episodes/points from the locked core_v1 cell plus the W5 exploratory cells
(A3, A3+, A6) and writes five PNGs into runs/exploratory/plots/.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from askback.eval.diagnostics import episode_blind_share

CORE = "runs/core_v1"
A3 = "runs/ablations/a3"
A3PLUS = "runs/cooperative_cell"
A6 = "runs/ablations/a6"
A6_P2LOW = "runs/ablations/a6_p2low"
OUT = "runs/exploratory/plots"


def load_jsonl(path):
    return [json.loads(line) for line in open(Path(path), encoding="utf-8")]


def points_by(run):
    rows = load_jsonl(run + "/points.jsonl")
    key = {}
    for r in rows:
        key[(r["policy"], r["persona"], r["stop_rule"], r["param"])] = r
    return key


def pass_curve(eps):
    """Fraction of episodes that pass at <= b turns (S0-style cumulative)."""
    turns = sorted(e["turns"] for e in eps if e["validity"])
    n = len(eps)
    curve = []
    for b in range(1, max(turns, default=1) + 1):
        done = sum(1 for t in turns if t <= b)
        curve.append((b, done / n))
    return curve


def p1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [
        ("realistic (confirmatory null)", CORE,
         ["P0", "P1", "P2", "P3"]),
        ("cooperative (exploratory)", None,
         ["P0", "P1", "P2", "P3"]),
    ]
    a3 = load_jsonl(A3 + "/episodes.jsonl")
    a3p = load_jsonl(A3PLUS + "/episodes.jsonl")
    coop = {p: [e for e in a3p if e["policy"] == p] for p in ("P0", "P1", "P2")}
    coop["P3"] = [e for e in a3 if e["policy"] == "P3"]
    for ax, (title, run, pols) in zip(axes, panels):
        src = load_jsonl(run + "/episodes.jsonl") if run else None
        for pol in pols:
            eps = [e for e in src if e["policy"] == pol] if src else coop[pol]
            curve = pass_curve(eps)
            if curve:
                xs, ys = zip(*curve)
                ax.plot(xs, ys, marker="o", ms=3, label=pol)
        ax.set_title(title)
        ax.set_xlabel("turn budget b")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("fraction passing at ≤ b")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT + "/p1_envelope_by_persona.png", dpi=150)
    plt.close(fig)


def p2():
    """Stop-rule savings S0 - S2(0.7) by policy x persona (mean turns)."""
    rows = []
    for run in (CORE, A3, A3PLUS):
        pts = points_by(run)
        for (pol, persona, rule, param), r in pts.items():
            if rule == "S0" or rule == "S2" and param == 0.7:
                rows.append((pol, persona, r))
    pols = ["P0", "P1", "P2", "P3"]
    pas = ["realistic", "cooperative"]
    save = {p: {pa: 0.0 for pa in pas} for p in pols}
    count = {p: {pa: 0 for pa in pas} for p in pols}
    s0 = {p: {pa: None for pa in pas} for p in pols}
    for pol, persona, r in rows:
        if persona not in pas:
            continue
        if r["stop_rule"] == "S0":
            s0[pol][persona] = r["mean_turns"]
        else:
            if s0[pol][persona] is not None:
                save[pol][persona] += s0[pol][persona] - r["mean_turns"]
                count[pol][persona] += 1
    import numpy as np
    x = np.arange(len(pols))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, pa in enumerate(pas):
        vals = [save[p][pa] / count[p][pa] if count[p][pa] else 0.0 for p in pols]
        ax.bar(x + (i - 0.5) * w, vals, w, label=pa)
    ax.set_xticks(x)
    ax.set_xticklabels(pols)
    ax.set_ylabel("mean turn savings S0 − S2(0.7)")
    ax.set_title("P2: stop-rule savings by policy × persona")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT + "/p2_stop_savings.png", dpi=150)
    plt.close(fig)


def p3():
    """Show S1 validity by density and cutoff; no T-quantile contrast is estimable."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    data = json.loads(Path("runs/exploratory/a6_s1.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharey=True)
    colors = {"P2": "#2a78d6", "P3": "#eb6834"}
    for row, c in enumerate(data["c_values"]):
        result = next(x for x in data["results"] if x["c"] == c)
        for col, density in enumerate(("high", "low")):
            ax = axes[row, col]
            values = [result["validity_rate"][f"{policy}_{density}"] for policy in ("P2", "P3")]
            bars = ax.bar(("P2", "P3"), values,
                          color=[colors["P2"], colors["P3"]], edgecolor="#222222")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                        f"{value:.0%}", ha="center", va="bottom", fontsize=9)
            ax.set_title(f"{density.capitalize()} density")
            ax.set_ylim(0, 0.12)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(True, axis="y", alpha=0.25)
            if col == 0:
                ax.set_ylabel(f"S1 validity rate\nc = {c:.2f}")
    fig.suptitle(
        "Two stop rules, two roles\n"
        "S0 — mechanism test (P3 ≈ P2; null, not degenerate) · "
        "S1(c) — untestable (calibration; 95–100% stop at T=0)",
        fontsize=12,
    )
    fig.text(0.5, 0.045, "T(v) unavailable at v = 0.10, 0.15, 0.20; no delta-in-delta or CI estimable.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.02, "Two stop rules, two roles. S0 exercises the mechanism test: P3's VoI-informed ordering produces identical mean turns to P2's uncertainty ordering (ΔT = −0.03) and 2 fewer turns than P0's fixed-order ordering. S1 at c ∈ {0.02, 0.05, 0.10} fires before any question in 95–100% of episodes — the grid requires calibration to per-cell ΔP before S1 produces interpretable operating points. 100 existing episodes per cell; no new runs. Axis capped at 12% for legibility.",
             ha="center", fontsize=9)
    fig.legend(handles=[Patch(facecolor=colors[p], edgecolor="#222222", label=p)
                        for p in ("P2", "P3")], loc="upper right", ncol=2,
               bbox_to_anchor=(0.98, 0.975), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(OUT + "/p3_density.png", dpi=150)
    plt.close(fig)


def p4():
    """Wasted-turn rate vs stop-rule × persona (mean across policies)."""
    import numpy as np
    agg = defaultdict(list)
    for run in (CORE, A3, A3PLUS, A6):
        for r in points_by(run).values():
            pa = r["persona"]
            if pa not in ("realistic", "cooperative"):
                continue
            agg[(pa, r["stop_rule"])].append(r["wasted_turn_rate"])
    rules = ["S0", "S1", "S2"]
    pas = ["realistic", "cooperative"]
    x = np.arange(len(rules))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, pa in enumerate(pas):
        vals = [np.mean(agg[(pa, r)]) if agg[(pa, r)] else 0.0 for r in rules]
        ax.bar(x + (i - 0.5) * w, vals, w, label=pa)
    ax.set_xticks(x)
    ax.set_xticklabels(rules)
    ax.set_ylabel("mean wasted-turn rate")
    ax.set_title("P4: wasted-turn rate vs stop-rule × persona")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT + "/p4_wasted_rate.png", dpi=150)
    plt.close(fig)


def p5():
    """Blind-spot targeting rate vs final validity scatter (per episode)."""
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (title, run) in zip(axes, [("realistic", CORE),
                                        ("cooperative", A3PLUS)]):
        xs, ys = [], []
        for e in load_jsonl(run + "/episodes.jsonl"):
            share, _, _ = episode_blind_share(e)
            xs.append(share)
            ys.append(1.0 if e["validity"] else 0.0)
        rng = np.random.default_rng(0)
        jy = np.array(ys) + rng.uniform(-0.04, 0.04, len(ys))
        ax.scatter(xs, jy, s=12, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("blind-spot targeting rate")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("final validity (jittered)")
    fig.tight_layout()
    fig.savefig(OUT + "/p5_blindspot_validity.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    Path(OUT).mkdir(parents=True, exist_ok=True)
    p1()
    p2()
    p3()
    p4()
    p5()
    print(f"wrote 5 plots to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
