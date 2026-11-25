"""Episode runner — PRD §9.1 (as amended by A10), §11, §9.3, §13.1, §15.1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from askback.belief.merge import DirectSetMerger
from askback.belief.state import Belief, from_extraction
from askback.domain.schema import Domain
from askback.domain.validator import PASS, validate
from askback.engine.sets import compute_sets
from askback.engine.voi import N_MC, compute_voi
from askback.extract.p2_adapter import P2Extractor
from askback.extract.stub import StubExtractor
from askback.policies.base import Policy
from askback.seeds import sim_rng, turn_rng, voi_rng
from askback.sim.answerer import REFUSE, AnalyticAnswerer, sample_knowledge_set
from askback.sim.personas import Persona
from askback.stops.rules import s1_fires, s2_fires, s2_tau_fires


class Simulator(Protocol):
    def answer(self, field: int, rng: np.random.Generator) -> int | str: ...


class Merger(Protocol):
    def merge(self, belief: Belief, field: int, value: int) -> tuple[Belief, bool]: ...


@dataclass(frozen=True)
class RunConfig:
    """Minimal episode config; the §14.1 YAML loader arrives with the CLI."""

    initial_nl: str = ""
    extractor: str = "stub"  # "stub" | "p2"; §5 refuses p2 at build time (V2)
    validator_tier: int = 1
    stop_rule: str = "S0"  # "S0" | "S1" | "S2" (§9.2, A10)
    s1_c: float = 0.02
    s2_tau: float = 1.0
    voi_n_mc: int = N_MC


@dataclass
class TurnRecord:
    """§9.3 offline-replay record, one per asked turn.

    ``delta_p`` is the pre-ask top-of-turn ``max_{f in U_{t+1}} ΔP_f`` computed
    just after turn ``t`` — the exact value §9.2 S1 compares with ``c`` at the
    next turn, which is the turn that can stop without producing a record of
    its own (A12/F5). Replay reads it there, so it reproduces S1 by
    construction. The two counts are what A10's S2(τ) needs; neither is
    recoverable from the booleans and the resolved fraction.
    """

    t: int
    field: int
    delta_p: float
    answer_type: str  # "value" | "refuse"
    validator_pass_after: bool
    resolved_fraction_after: float
    g_empty_after: bool
    u_empty_after: bool
    determinism_hash: str
    resolvable_count_after: int = 0
    u_count_after: int = 0


@dataclass(frozen=True)
class InitialState:
    """The pre-turn state §9.1 judges at t = 0 (A12, Behavior B).

    With Behavior B a rule stop can end an episode before any turn executes,
    leaving no records for replay to read. This carries the one state that
    replay needs to reproduce that decision.
    """

    u: frozenset[int]
    g: frozenset[int]
    delta_p: float  # 0.0 unless the rule stop needed a turn-0 VoI
    validator_pass: bool


@dataclass
class EpisodeResult:
    records: list[TurnRecord] = field(default_factory=list)
    initial: InitialState | None = None
    trajectory_hash: str = ""  # §16 run-level fingerprint (A12/R4a)
    turns: int = 0
    passed: bool = False
    abandoned: bool = False
    reason: str = ""
    validity: bool = False
    refusals: int = 0
    conflicts: int = 0
    redundant_questions: int = 0
    refused_count: int = 0  # A13: unique fields refused this episode
    belief: Belief | None = None


# §15.1's return type. The PRD never defines it; W1's result dataclass is it.
Trajectory = EpisodeResult


def _hash(belief: Belief) -> str:
    """Per-turn state fingerprint on `TurnRecord` (A12/R4a)."""
    h = hashlib.sha256()
    for p in belief.probs:
        h.update(p.astype(np.float64).tobytes())
    return h.hexdigest()[:16]


def _trajectory_hash(records: list[TurnRecord], seed: np.random.SeedSequence,
                     cfg: dict) -> str:
    """§16: `sha256(config_canonical_json + episode_seed + trajectory_json)`."""
    h = hashlib.sha256()
    h.update(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode())
    h.update(b"|" + str(seed.entropy).encode() + b"|")
    h.update(json.dumps([asdict(r) for r in records],
                        sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def _rule_fires(
    rule: str,
    delta_p: dict[int, float] | None,
    u: set[int],
    g: set[int],
    s1_c: float,
    s2_tau: float,
) -> bool:
    """§9.1 #3: reads the current (U, G) and the current turn's ΔP.

    A12 Behavior B: at t = 0 the current state is the initial post-extraction
    state and ΔP is computed for it, so a rule stop can fire at T = 0.
    """
    if delta_p is None or rule == "S0":
        return False
    if rule == "S1":
        return s1_fires(delta_p, s1_c)
    if rule == "S2":
        if s2_tau >= 1.0:
            return s2_fires(g)
        return s2_tau_fires(g, u, s2_tau)
    raise ValueError(f"unknown stop rule: {rule!r}")


def _run_episode(
    domain: Domain,
    belief: Belief,
    policy: Policy,
    simulator: Simulator,
    merger: Merger,
    seed: np.random.SeedSequence,
    *,
    k: int = 1,
    patience: int = 20,
    stop_rule: str = "S0",
    s1_c: float = 0.02,
    s2_tau: float = 1.0,
    voi_n_mc: int = N_MC,
    cfg_extra: dict | None = None,
) -> EpisodeResult:
    """Run one episode to its terminal condition. FR-11 bound is asserted.

    Internal helper; the public §15.1 entry point is :func:`run_episode`.
    """
    n = domain.n_fields
    bound = min(patience, n + 1)
    result = EpisodeResult(belief=belief)
    t = 0
    refused: set[int] = set()  # A13: fields the simulator refused this episode

    U, R, G = compute_sets(belief, domain, k, refused)
    ok, _ = validate(belief.probs, domain, k)

    # A12 Behavior B: t = 0 is judged on the initial post-extraction state. A
    # rule stop needs ΔP there; compute it once and reuse it for selection so
    # VoI still costs one call per executed turn.
    delta_p: dict[int, float] | None = None
    if stop_rule != "S0":
        delta_p = compute_voi(belief, domain, k, voi_rng(seed, 0), voi_n_mc)
    result.initial = InitialState(
        u=frozenset(U),
        g=frozenset(G),
        delta_p=max(delta_p.values(), default=0.0) if delta_p else 0.0,
        validator_pass=ok,
    )

    while True:
        assert t <= bound, f"FR-11 violated: t={t} > {bound}"

        if t >= patience:  # §9.1 #1
            result.reason, result.abandoned = "patience_exhausted", True
            break
        if not U:  # §9.1 #2
            result.reason, result.passed = "u_empty", True
            break
        # A12/F5: S1/S2 judge the top-of-turn ΔP, which the previous turn's
        # record already carries (computed post-merge, below).
        if _rule_fires(stop_rule, delta_p, U, G, s1_c, s2_tau):  # §9.1 #3
            result.reason = "rule_stop"
            break
        if ok:  # §9.1 #4
            result.reason, result.passed = "validator_pass", True
            break

        if delta_p is None:  # S0 computes it only once a question is certain
            delta_p = compute_voi(belief, domain, k, voi_rng(seed, t), voi_n_mc)
        f = policy.select(U, delta_p, belief, turn_rng(seed, t))
        if f in R:
            result.redundant_questions += 1

        answer = simulator.answer(f, sim_rng(seed, t))
        if answer == REFUSE:
            answer_type = "refuse"
            result.refusals += 1
            refused.add(f)  # A13: a refused field leaves U for the rest of the episode
        else:
            answer_type = "value"
            belief, conflict = merger.merge(belief, f, int(answer))
            if conflict:
                result.conflicts += 1

        U, R, G = compute_sets(belief, domain, k, refused)  # A13: refused leaves U
        ok, _ = validate(belief.probs, domain, k)

        # A12/F5: a rule stop asks nothing, so the turn that stops owns no
        # record; the ΔP S1 reads at t + 1 lands on this record instead. Same
        # computation, same seed as the next turn's top-of-turn VoI, so cost
        # is unchanged and the next turn reuses it. S0 is untouched: it logs
        # the ΔP its selection used, over the post-turn U, as it always has.
        if stop_rule == "S0":
            logged_delta = max((delta_p[g] for g in U), default=0.0)
            delta_p = None
        else:
            delta_p = compute_voi(
                belief, domain, k, voi_rng(seed, t + 1), voi_n_mc
            )
            logged_delta = max(delta_p.values(), default=0.0)

        result.records.append(
            TurnRecord(
                t=t,
                field=f,
                delta_p=logged_delta,
                answer_type=answer_type,
                validator_pass_after=ok,
                resolved_fraction_after=(n - len(U)) / n,
                g_empty_after=not (U - R),
                u_empty_after=not U,
                determinism_hash=_hash(belief),
                resolvable_count_after=len(R),
                u_count_after=len(U),
            )
        )
        t += 1

    assert t <= bound, f"FR-11 violated at exit: t={t} > {bound}"
    result.turns = t
    result.belief = belief
    result.validity = ok
    result.refused_count = len(refused)
    result.trajectory_hash = _trajectory_hash(result.records, seed, {
        "k": k, "patience": patience, "stop_rule": stop_rule,
        "s1_c": s1_c, "s2_tau": s2_tau, "voi_n_mc": voi_n_mc,
        **(cfg_extra or {}),
    })
    return result


def run_episode(
    cfg: RunConfig,
    domain: Domain,
    policy: Policy,
    persona: Persona,
    seed: np.random.SeedSequence,
) -> Trajectory:
    """§15.1 entry point: extract, build the answerer, run the episode.

    `persona` is owned here, never by `_run_episode` (A9).
    """
    if cfg.extractor == "p2":
        P2Extractor()  # raises: no P2 artifacts at build time (PRD §5, A12 V2)
    if cfg.extractor != "stub":
        raise ValueError(f"unknown extractor: {cfg.extractor!r}")
    extraction = StubExtractor(domain, turn_rng(seed, 0)).extract(cfg.initial_nl)
    belief = from_extraction(domain, extraction)
    knowledge = sample_knowledge_set(domain, persona.rho, seed)
    return _run_episode(
        domain,
        belief,
        policy,
        AnalyticAnswerer(persona, domain, knowledge),
        DirectSetMerger(),
        seed,
        k=cfg.validator_tier,
        patience=persona.patience,
        stop_rule=cfg.stop_rule,
        s1_c=cfg.s1_c,
        s2_tau=cfg.s2_tau,
        voi_n_mc=cfg.voi_n_mc,
        cfg_extra=asdict(cfg),
    )


if __name__ == "__main__":
    from askback.belief.state import initial_belief
    from askback.domain.generator import generate_domain
    from askback.policies.p0 import P0FixedOrder
    from askback.sim.personas import PERSONAS

    dom = generate_domain(20250, "low")
    seed = np.random.SeedSequence(7)

    class TruthSim:
        def __init__(self, domain):
            self.domain = domain

        def answer(self, field, rng):
            return self.domain.truth[field]

    res = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), TruthSim(dom), DirectSetMerger(),
        seed, k=2,
    )
    assert res.passed and not res.abandoned, res.reason
    assert res.validity and res.turns <= dom.n_fields
    assert all(r.u_count_after >= r.resolvable_count_after for r in res.records)

    class LazySim:
        def answer(self, field, rng):
            return REFUSE

    res2 = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), LazySim(), DirectSetMerger(),
        seed, patience=3,
    )
    assert res2.abandoned and res2.turns == 3 and res2.refusals == 3
    assert not res2.validity

    wrapped = run_episode(RunConfig(validator_tier=2), dom, P0FixedOrder(), PERSONAS["cooperative"], seed)
    assert wrapped.passed and wrapped.validity, wrapped.reason
    print("episode ok", res.turns, res2.turns, wrapped.turns)