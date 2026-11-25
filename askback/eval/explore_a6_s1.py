"""A6 S1 counterfactual replay on the existing A1 episodes; generates no episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from askback.belief.merge import DirectSetMerger
from askback.belief.state import from_extraction
from askback.domain.generator import generate_domain
from askback.domain.validator import validate
from askback.engine.sets import compute_sets
from askback.engine.voi import compute_voi
from askback.eval.stats import SEED, quantile
from askback.runner.episode import InitialState
from askback.seeds import sim_rng, voi_rng
from askback.sim.answerer import REFUSE, AnalyticAnswerer, sample_knowledge_set
from askback.sim.personas import PERSONAS

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("runs/core_v1", "runs/ablations/a6", "runs/ablations/a6_p2low")
POLICY_DENSITY = (("P2", "high"), ("P2", "low"),
                  ("P3", "high"), ("P3", "low"))
C_VALUES = (0.02, 0.05, 0.10)
V_VALUES = (0.10, 0.15, 0.20)
N_RESAMPLES = 10_000


def load_counterfactual_rows(run_dir: Path) -> dict[tuple[str, str], list[tuple[dict, InitialState, list[tuple[float, bool, bool]]]]]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    domain = generate_domain(cfg["master_seed"], cfg["domain"]["rule_density"])
    groups: dict[tuple[str, str], list] = {}
    source = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    for line in source:
        row = json.loads(line)
        if row["policy"] not in ("P2", "P3"):
            continue
        seed = np.random.SeedSequence(
            row["seed"]["entropy"], spawn_key=tuple(row["seed"]["spawn_key"])
        )
        belief = from_extraction(domain, {
            x["field"]: (x["value"], x["confidence"])
            for x in row["extraction"]
        })
        k = row["validator_tier"]
        refused: set[int] = set()
        initial_u, _, _ = compute_sets(belief, domain, k, refused)
        initial_dp = compute_voi(
            belief, domain, k, voi_rng(seed, 0), cfg["voi"]["n_mc"]
        )
        initial = InitialState(
            frozenset(initial_u), frozenset(row["initial"]["g"]),
            max(initial_dp.values(), default=0.0), row["initial"]["validator_pass"],
        )
        if initial.u != frozenset(row["initial"]["u"]):
            raise AssertionError("reconstructed initial U differs from stored U")

        answerer = AnalyticAnswerer(
            PERSONAS[row["persona"]], domain,
            sample_knowledge_set(domain, PERSONAS[row["persona"]].rho, seed),
        )
        path = []
        for raw in row["records"]:
            field = raw["field"]
            answer = answerer.answer(field, sim_rng(seed, raw["t"]))
            answer_type = "refuse" if answer == REFUSE else "value"
            if answer_type != raw["answer_type"]:
                raise AssertionError("stored answer type differs from deterministic replay")
            if answer == REFUSE:
                refused.add(field)
            else:
                belief, _ = DirectSetMerger().merge(belief, field, int(answer))

            u, _, _ = compute_sets(belief, domain, k, refused)
            valid, _ = validate(belief.probs, domain, k)
            if len(u) != raw["u_count_after"] or valid != raw["validator_pass_after"]:
                raise AssertionError("reconstructed post-turn state differs from log")
            post_dp = compute_voi(
                belief, domain, k, voi_rng(seed, raw["t"] + 1), cfg["voi"]["n_mc"]
            )
            path.append((max((post_dp[f] for f in u), default=0.0), not u, valid))
        groups.setdefault((row["policy"], row["density"]), []).append((row, initial, path))
    return groups


def replay_cell(rows, c: float, v: float, indices=None):
    selected = range(len(rows)) if indices is None else indices
    turns, validities = [], []
    for i in selected:
        row, initial, path = rows[int(i)]
        turns_used, valid = 0, initial.validator_pass
        if initial.u and initial.delta_p >= c and not initial.validator_pass:
            for delta_p, u_empty, pass_after in path:
                turns_used += 1
                # Match replay(): patience, U empty, S1 threshold, validator.
                if turns_used >= row["patience"]:
                    valid = pass_after
                    break
                if u_empty or delta_p < c or pass_after:
                    valid = pass_after
                    break
            else:
                valid = False
        turns.append(turns_used)
        validities.append(valid)
    q = quantile(turns, validities, v, total=len(turns))
    return q, sum(validities) / len(validities)


def analyze() -> dict:
    groups = {}
    for run in RUNS:
        groups.update(load_counterfactual_rows(ROOT / run))
    if set(groups) != set(POLICY_DENSITY) or any(len(groups[g]) != 100 for g in POLICY_DENSITY):
        raise AssertionError("A6 requires four 100-episode policy × density cells")

    rng = np.random.default_rng(SEED)
    results = []
    for c in C_VALUES:
        for v in V_VALUES:
            quantiles, rates = {}, {}
            for group in POLICY_DENSITY:
                q, rate = replay_cell(groups[group], c, v)
                quantiles[f"{group[0]}_{group[1]}"] = q
                rates[f"{group[0]}_{group[1]}"] = rate

            names = ("P2_high", "P2_low", "P3_high", "P3_low")
            q = [quantiles[name] for name in names]
            observed = None if any(x is None for x in q) else (q[2] - q[0]) - (q[3] - q[1])
            boot = []
            if observed is not None:
                for _ in range(N_RESAMPLES):
                    sampled = {}
                    for group, name in zip(POLICY_DENSITY, names):
                        ids = rng.integers(0, len(groups[group]), size=len(groups[group]))
                        sampled[name] = replay_cell(groups[group], c, v, ids)[0]
                    if all(x is not None for x in sampled.values()):
                        boot.append((sampled["P3_high"] - sampled["P2_high"])
                                    - (sampled["P3_low"] - sampled["P2_low"]))
            ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])] if boot else None
            results.append({
                "c": c, "v": v, "T": quantiles, "validity_rate": rates,
                "delta_in_delta_turns": observed, "bootstrap_95ci": ci,
                "valid_bootstrap_resamples": len(boot),
            })

    return {
        "method": (
            "A6 B4: counterfactual S1 replay on existing S0 episodes. Initial and "
            "post-turn ΔP reconstructed deterministically from stored extraction, "
            "seed, and answer trajectory; no episodes generated. Replay applies "
            "patience, empty-U, S1, then validator ordering. Independent-groups "
            "percentile bootstrap; density_id changes seed stream, so groups cannot "
            "be paired."
        ),
        "source_runs": list(RUNS), "policies": ["P2", "P3"],
        "densities": ["high", "low"], "c_values": list(C_VALUES),
        "v_values": list(V_VALUES), "n_episodes_per_cell": 100,
        "bootstrap": {"method": "independent-groups percentile", "resamples": N_RESAMPLES,
                      "seed": SEED, "confidence": 0.95},
        "results": results,
    }


if __name__ == "__main__":
    out = ROOT / "runs/exploratory/a6_s1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analyze(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
