"""T-08: 10-seed skeleton run — PRD §18. Kept as the W2 regression."""

from __future__ import annotations

import numpy as np
import pytest

from askback.belief.merge import DirectSetMerger
from askback.belief.state import initial_belief
from askback.domain.generator import generate_domain
from askback.domain.schema import Domain, Rule
from askback.domain.validator import PASS, validate
from askback.policies.p0 import P0FixedOrder
from askback.runner.episode import REFUSE, RunConfig, _run_episode, run_episode
from askback.sim.personas import PERSONAS


class TruthSim:
    def __init__(self, domain):
        self.domain = domain

    def answer(self, field, rng):
        return self.domain.truth[field]


class LazySim:
    def answer(self, field, rng):
        return REFUSE


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("density", ["low", "high"])
def test_skeleton_run(seed: int, density: str) -> None:
    dom = generate_domain(seed, density)
    rng_seed = np.random.SeedSequence(seed)

    res = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), TruthSim(dom), DirectSetMerger(),
        rng_seed, k=2,
    )
    assert res.passed and not res.abandoned, res.reason
    # Propagation can resolve a field before it is asked, so turns may be < 15.
    assert 1 <= res.turns <= dom.n_fields
    # §9.1 #4: the validator passes on its own tier-k closure, which can leave a
    # propagated field unasked and still ⊥ in the belief.
    if res.reason == "u_empty":
        assert res.records[-1].resolved_fraction_after == 1.0
        assert res.records[-1].u_empty_after
    else:
        assert res.reason == "validator_pass"
        assert validate(res.belief.probs, dom, 2) == (True, PASS)
    assert res.validity
    # A11: delta_p is a real ΔP now; §7.2 bounds it to [-1, 1].
    assert all(r.answer_type == "value" and -1.0 <= r.delta_p <= 1.0 for r in res.records)
    assert all(r.u_count_after >= r.resolvable_count_after for r in res.records)
    assert res.refusals == 0 and res.conflicts == 0

    again = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), TruthSim(dom), DirectSetMerger(),
        np.random.SeedSequence(seed), k=2,
    )
    assert [r.determinism_hash for r in again.records] == [
        r.determinism_hash for r in res.records
    ]
    # V3: the hash is populated on every record, not left empty.
    assert all(len(r.determinism_hash) == 16 for r in res.records)
    # A12/R4a: §16 run-level fingerprint, equal for same seed + config.
    assert again.trajectory_hash == res.trajectory_hash
    assert len(res.trajectory_hash) == 64

    lazy = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), LazySim(), DirectSetMerger(),
        rng_seed, patience=3,
    )
    assert lazy.abandoned and lazy.reason == "patience_exhausted"
    assert lazy.turns == 3 and len(lazy.records) == 3
    assert lazy.refusals == 3
    assert all(r.answer_type == "refuse" for r in lazy.records)


@pytest.mark.parametrize("density", ["low", "high"])
def test_wrapper_runs_end_to_end(density: str) -> None:
    """A9 wrapper: extraction, knowledge set, answerer, §13.1 counters."""
    dom = generate_domain(20250, density)
    res = run_episode(
        RunConfig(validator_tier=2),
        dom,
        P0FixedOrder(),
        PERSONAS["cooperative"],
        np.random.SeedSequence(1),
    )
    assert res.passed and not res.abandoned, res.reason
    assert res.validity
    assert res.turns <= dom.n_fields
    assert res.redundant_questions <= res.turns

    again = run_episode(
        RunConfig(validator_tier=2),
        dom,
        P0FixedOrder(),
        PERSONAS["cooperative"],
        np.random.SeedSequence(1),
    )
    assert [r.determinism_hash for r in again.records] == [
        r.determinism_hash for r in res.records
    ]
    assert all(len(r.determinism_hash) == 16 for r in res.records)
    assert again.trajectory_hash == res.trajectory_hash
    # Different episode seed ⇒ different run fingerprint.
    other = run_episode(
        RunConfig(validator_tier=2), dom, P0FixedOrder(),
        PERSONAS["cooperative"], np.random.SeedSequence(2),
    )
    assert other.trajectory_hash != res.trajectory_hash


def test_wrapper_refuses_p2_extractor() -> None:
    """V2: §5 — no P2 artifacts at build time, so the wrapper must refuse."""
    dom = generate_domain(20250, "low")
    with pytest.raises(NotImplementedError):
        run_episode(
            RunConfig(extractor="p2"),
            dom,
            P0FixedOrder(),
            PERSONAS["cooperative"],
            np.random.SeedSequence(1),
        )


def test_wrapper_patience_comes_from_persona() -> None:
    dom = generate_domain(20250, "low")
    res = run_episode(
        RunConfig(validator_tier=2),
        dom,
        P0FixedOrder(),
        PERSONAS["impatient_expert"],
        np.random.SeedSequence(3),
    )
    assert res.turns <= PERSONAS["impatient_expert"].patience
    if res.abandoned:
        assert res.reason == "patience_exhausted"
        assert res.turns == PERSONAS["impatient_expert"].patience


def test_p0_no_repeat_on_refusal() -> None:
    """A13: field 0 is outside the knowledge set, so the simulator REFUSEs it;
    P0 must never ask it again. Five turns max, field 0 exactly once.
    """
    dom = Domain(
        domains=[[0, 1]] * 5,
        rules=[],
        truth=[0] * 5,
    )

    class RefuseZeroSim:
        def __init__(self, domain):
            self.domain = domain

        def answer(self, field, rng):
            if field == 0:  # outside knowledge_set
                return REFUSE
            return self.domain.truth[field]

    res = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), RefuseZeroSim(dom),
        DirectSetMerger(), np.random.SeedSequence(0), k=1, patience=5,
    )
    assert res.refusals == 1 and res.refused_count == 1
    assert res.records[0].field == 0
    assert all(r.field != 0 for r in res.records[1:])
    assert res.turns == 5
    assert res.reason == "patience_exhausted" and res.abandoned


def test_validity_uses_validator_verdict_not_completion() -> None:
    """V5: §13.2 validity must be the validator verdict (`validator_pass_after[-1]`),
    not `u_empty_after[-1]` or `passed`. Fixture: U=∅ with a rule violation —
    the episode completes (u_empty) but validity is False, and equals the last
    record's validator verdict. If validity read `u_empty_after[-1]`, S0 would
    show 100% validity, contradicting the observed ~12.8% ceiling.
    """
    # Two conflicting rules on field 2, both with premises below it. Resolving
    # field 0 and field 1 fires both, so a violation is detected on the final
    # validate even though every field ends up resolved (U = ∅).
    dom = Domain(
        domains=[[0, 1]] * 4,
        rules=[
            Rule((0,), (0,), 2, 0),
            Rule((1,), (0,), 2, 1),
        ],
        truth=[0, 0, 0, 0],
    )

    res = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), TruthSim(dom),
        DirectSetMerger(), np.random.SeedSequence(0), k=1, patience=10,
    )
    assert res.reason == "u_empty", res.reason
    assert res.records[-1].u_empty_after
    assert not res.records[-1].validator_pass_after
    # validity tracks the validator verdict, not completion / u_empty.
    assert not res.validity
    assert res.validity == res.records[-1].validator_pass_after