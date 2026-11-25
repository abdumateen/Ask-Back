"""§14 config loading, A1 episode execution, A2 replay aggregation.

The §14.2 entry points (`askback.run`, `askback.replay`) are thin wrappers over
this module. Everything here is deterministic: rows are written sorted with
canonical JSON, so FR-16's byte-identical reruns hold for both files.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from askback.domain.generator import generate_domain
from askback.engine.voi import EXACT_MAX_JOINT, EXACT_MAX_UNCERTAIN, N_MC
from askback.extract.stub import P_CLEAN, P_MISSING, P_UNCERTAIN, StubExtractor
from askback.policies.p0 import P0FixedOrder
from askback.policies.p1 import P1Random
from askback.policies.p2 import P2Uncertainty
from askback.policies.p3 import P3VoI
from askback.runner.episode import (
    InitialState,
    RunConfig,
    TurnRecord,
    run_episode,
)
from askback.runner.oracle import episode_bounds
from askback.seeds import episode_seed, turn_rng
from askback.sim.answerer import sample_knowledge_set
from askback.sim.personas import PERSONAS
from askback.stops.replay import replay

POLICY_CLASSES = {
    "P0": P0FixedOrder,
    "P1": P1Random,
    "P2": P2Uncertainty,
    "P3": P3VoI,
}
# §16's seed tuple needs stable small ints; these orders are the registry order
# of `sim/personas.py` and the two densities the generator implements.
POLICY_IDS = {name: i for i, name in enumerate(POLICY_CLASSES)}
PERSONA_IDS = {name: i for i, name in enumerate(PERSONAS)}
DENSITY_IDS = {"low": 0, "high": 1}

N_FIELDS = 15  # `domain/generator.py` fixes the generator's field count


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} is not a mapping")
    return cfg


def _need(cfg: dict, *keys):
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"config missing {'.'.join(keys)}")
        node = node[key]
    return node


def validate_config(cfg: dict) -> None:
    """Config ≈ code parity. Any drift is an error, never a silent ignore."""
    if _need(cfg, "domain", "n_fields") != N_FIELDS:
        raise ValueError(f"domain.n_fields must be {N_FIELDS}")
    if _need(cfg, "domain", "rule_density") not in DENSITY_IDS:
        raise ValueError("domain.rule_density must be 'low' or 'high'")
    if _need(cfg, "extractor", "type") != "stub":
        raise ValueError("extractor.type must be 'stub' (§5: p2 refused at build)")
    for key, want in (
        ("p_clean", P_CLEAN),
        ("p_uncertain", P_UNCERTAIN),
        ("p_missing", P_MISSING),
    ):
        if float(_need(cfg, "extractor", key)) != want:
            raise ValueError(f"extractor.{key} must be {want}")
    if _need(cfg, "merger") != "direct":
        raise ValueError("merger must be 'direct'")
    if _need(cfg, "simulator", "answer_renderer") != "analytic":
        raise ValueError("simulator.answer_renderer must be 'analytic'")
    for key, want in (
        ("n_mc", N_MC),
        ("exact_max_uncertain", EXACT_MAX_UNCERTAIN),
        ("exact_max_states", EXACT_MAX_JOINT),
    ):
        if int(_need(cfg, "voi", key)) != want:
            raise ValueError(f"voi.{key} must be {want} (A12 F3/R1)")
    policies = _need(cfg, "policies")
    if not policies or any(p not in POLICY_CLASSES for p in policies):
        raise ValueError(f"policies must be a subset of {sorted(POLICY_CLASSES)}")
    personas = _need(cfg, "personas")
    if not personas or any(p not in PERSONAS for p in personas):
        raise ValueError(f"personas must be a subset of {sorted(PERSONAS)}")
    if int(_need(cfg, "n_episodes")) < 1:
        raise ValueError("n_episodes must be >= 1")
    grid = _need(cfg, "stop_grids", "S1_c")
    if len(grid) != 5 or any(not 0 < float(c) < 1 for c in grid):
        raise ValueError("stop_grids.S1_c must be five probabilities in (0, 1)")


def effective_config(
    cfg: dict, n_episodes: int | None = None, master_seed: int | None = None
) -> dict:
    eff = json.loads(json.dumps(cfg))  # plain deep copy, YAML-free
    if n_episodes is not None:
        eff["n_episodes"] = int(n_episodes)
    if master_seed is not None:
        eff["master_seed"] = int(master_seed)
    return eff


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _initial_dict(state: InitialState | None) -> dict | None:
    if state is None:
        return None
    return {
        "u": sorted(state.u),
        "g": sorted(state.g),
        "delta_p": state.delta_p,
        "validator_pass": state.validator_pass,
    }


def _episode_row(
    eff: dict, cfg_hash: str, policy: str, persona_name: str, idx: int,
    domain, seed: np.random.SeedSequence, res,
) -> dict:
    persona = PERSONAS[persona_name]
    extraction = StubExtractor(domain, turn_rng(seed, 0)).extract("")
    bounds = episode_bounds(
        domain, extraction,
        sample_knowledge_set(domain, persona.rho, seed),
    )
    return {
        "run_id": eff["run_id"],
        "config_hash": cfg_hash,
        "policy": policy,
        "persona": persona_name,
        "density": eff["domain"]["rule_density"],
        "episode_idx": idx,
        "policy_id": POLICY_IDS[policy],
        "persona_id": PERSONA_IDS[persona_name],
        "density_id": DENSITY_IDS[eff["domain"]["rule_density"]],
        "validator_tier": eff["domain"]["validator_tier"],
        "patience": persona.patience,
        "seed": {"entropy": seed.entropy, "spawn_key": list(seed.spawn_key)},
        "turns": res.turns,
        "validity": res.validity,
        "abandoned": res.abandoned,
        "reason": res.reason,
        "refusals": res.refusals,
        "conflicts": res.conflicts,
        "redundant_questions": res.redundant_questions,
        "bound_oracle_questions": bounds["oracle_questions"],
        "bound_no_interaction_pass": bounds["no_interaction"],
        "trajectory_hash": res.trajectory_hash,
        "initial": _initial_dict(res.initial),
        "records": [asdict(r) for r in res.records],
        "extraction": [
            {"field": f, "value": v, "confidence": c}
            for f, (v, c) in sorted(extraction.items())
        ],
    }


def run_study(cfg: dict, out_dir: str | Path) -> Path:
    """Execute A1's cell grid (S0 online) into ``out_dir/episodes.jsonl``."""
    validate_config(cfg)
    ch = config_hash(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    snapshot = out / "config.json"
    if snapshot.exists():
        old = json.loads(snapshot.read_text(encoding="utf-8"))
        if config_hash(old) != ch:
            raise ValueError(
                f"{snapshot} holds a different config; use a fresh --out"
            )
    else:
        snapshot.write_text(
            json.dumps(cfg, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    episodes_path = out / "episodes.jsonl"
    rows = _read_jsonl(episodes_path)
    seen = {
        (r["config_hash"], r["policy"], r["persona"], r["density"],
         r["episode_idx"])
        for r in rows
    }

    density = cfg["domain"]["rule_density"]
    domain = generate_domain(int(cfg["master_seed"]), density)
    k = int(cfg["domain"]["validator_tier"])

    for policy_name in cfg["policies"]:
        for persona_name in cfg["personas"]:
            for idx in range(int(cfg["n_episodes"])):
                key = (ch, policy_name, persona_name, density, idx)
                if key in seen:
                    continue
                seed = episode_seed(
                    int(cfg["master_seed"]),
                    POLICY_IDS[policy_name],
                    PERSONA_IDS[persona_name],
                    DENSITY_IDS[density],
                    idx,
                )
                res = run_episode(
                    RunConfig(validator_tier=k),
                    domain,
                    POLICY_CLASSES[policy_name](),
                    PERSONAS[persona_name],
                    seed,
                )
                rows.append(
                    _episode_row(cfg, ch, policy_name, persona_name, idx,
                                 domain, seed, res)
                )
                seen.add(key)

    rows.sort(
        key=lambda r: (r["policy_id"], r["persona_id"], r["density_id"],
                       r["episode_idx"])
    )
    _write_jsonl(episodes_path, rows)
    return episodes_path


def stop_points(cfg: dict) -> list[tuple[str, float, float, float | None]]:
    """A2's 8 points per policy: S0, S1 × 5, S2 × {0.7, 1.0} (A10/A12).

    ``param`` is ``None`` for ``"S0"``, the ``c`` value for ``"S1"``, and the
    ``tau`` value for ``"S2"``. ``(policy, stop_rule, param)`` is unique
    within a ``run_id``.
    """
    points = [("S0", 0.02, 1.0, None)]
    points += [("S1", float(c), 1.0, float(c)) for c in cfg["stop_grids"]["S1_c"]]
    points += [("S2", 0.02, tau, float(tau)) for tau in (0.7, 1.0)]
    return points


_RULE_ORDER = {"S0": 0, "S1": 1, "S2": 2}

# §13.2 `too_early`: a rule stop is premature when the remaining fields could
# still have moved the pass probability by at least this much, with patience
# left over.
TOO_EARLY_DELTA = 0.05


def _point_aggregates(outs: list[tuple[dict, object]]) -> dict:
    """§13.2 counters for one (policy, rule, param) point.

    ``redundant_questions`` is a trajectory property, so it comes from the
    episode row; refusals are counted inside the replayed prefix, since a
    truncated trajectory never asks the later turns.
    """
    n = len(outs)
    turns_total = sum(o.turns for _, o in outs)
    refusals = sum(
        sum(1 for r in row["records"][: o.turns] if r["answer_type"] == "refuse")
        for row, o in outs
    )
    too_early = too_late = 0
    for row, o in outs:
        if o.reason == "rule_stop" and row["patience"] - o.turns > 0:
            if o.turns:
                dp = row["records"][o.turns - 1]["delta_p"]
            else:
                dp = (row["initial"] or {}).get("delta_p", 0.0)
            too_early += dp >= TOO_EARLY_DELTA
        too_late += row["redundant_questions"] >= 1
    return {
        "redundant_question_rate": sum(r["redundant_questions"] for r, _ in outs)
        / max(turns_total, 1),
        "wasted_turn_rate": refusals / max(turns_total, 1),
        "stopping_quality": {"too_early": too_early / n, "too_late": too_late / n},
    }


def replay_study(run_dir: str | Path) -> Path:
    """Offline-replay every A2 point into ``run_dir/points.jsonl``."""
    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(run / "episodes.jsonl")
    if not rows:
        raise ValueError(f"{run / 'episodes.jsonl'} has no episodes; run first")

    points = stop_points(cfg)
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        init = InitialState(
            u=frozenset(row["initial"]["u"]),
            g=frozenset(row["initial"]["g"]),
            delta_p=row["initial"]["delta_p"],
            validator_pass=row["initial"]["validator_pass"],
        )
        records = [TurnRecord(**r) for r in row["records"]]
        for rule, c, tau, param in points:
            out = replay(records, rule=rule, c=c, tau=tau,
                         patience=row["patience"], initial=init)
            groups[(row["policy"], row["persona"], row["density"],
                    rule, param)].append((row, out))

    point_rows = []
    for (policy, persona, density, rule, param), outs in groups.items():
        n = len(outs)
        point_rows.append({
            "run_id": cfg["run_id"],
            "config_hash": config_hash(cfg),
            "policy": policy,
            "persona": persona,
            "density": density,
            "stop_rule": rule,
            "param": param,
            "n": n,
            "mean_turns": sum(o.turns for _, o in outs) / n,
            "mean_validity": sum(o.validity for _, o in outs) / n,
            "abandonment_rate": sum(o.abandoned for _, o in outs) / n,
            **_point_aggregates(outs),
        })
    point_rows.sort(
        key=lambda r: (POLICY_IDS[r["policy"]], _RULE_ORDER[r["stop_rule"]],
                       r["param"] if r["param"] is not None else -1.0)
    )
    keys = [
        (r["policy"], r["persona"], r["density"], r["stop_rule"], r["param"])
        for r in point_rows
    ]
    if len(keys) != len(set(keys)):
        raise AssertionError(
            "(policy, persona, density, stop_rule, param) is not unique"
        )
    expected = len(cfg["policies"]) * len(cfg["personas"]) * 8
    if len(point_rows) != expected:
        raise AssertionError(
            f"expected {expected} operating points, got {len(point_rows)}"
        )
    path = run / "points.jsonl"
    _write_jsonl(path, point_rows)
    return path


if __name__ == "__main__":
    import tempfile

    cfg = {
        "run_id": "selfcheck",
        "master_seed": 20250,
        "domain": {"n_fields": 15, "rule_density": "high",
                   "validator_tier": 1},
        "extractor": {"type": "stub", "p_clean": P_CLEAN,
                      "p_uncertain": P_UNCERTAIN, "p_missing": P_MISSING},
        "merger": "direct",
        "simulator": {"answer_renderer": "analytic"},
        "policies": ["P0", "P3"],
        "personas": ["realistic"],
        "n_episodes": 2,
        "voi": {"n_mc": N_MC, "exact_max_uncertain": EXACT_MAX_UNCERTAIN,
                "exact_max_states": EXACT_MAX_JOINT},
        "stop_grids": {"S1_c": [0.005, 0.01, 0.02, 0.05, 0.10]},
        "cost": {"alpha": 1.0, "beta": 10.0, "gamma": 5.0},
        "bootstrap": {"n_resamples": 10000, "ci": 0.95},
        "out_dir": "unused",
    }
    with tempfile.TemporaryDirectory() as tmp:
        eps = run_study(cfg, tmp)
        pts = replay_study(tmp)
        first = eps.read_bytes()
        run_study(cfg, tmp)  # resume: nothing new, same bytes
        assert eps.read_bytes() == first
        assert len(_read_jsonl(pts)) == 2 * 8
        assert len(_read_jsonl(eps)) == 4
    print("cli ok")