"""§9.2 rule stops (A10), §9.3 replay, T-09 — PRD §18."""

from __future__ import annotations

import numpy as np
import pytest

from askback.belief.merge import DirectSetMerger
from askback.belief.state import initial_belief
from askback.domain.generator import generate_domain
from askback.domain.schema import Domain, Rule
from askback.policies.p0 import P0FixedOrder
from askback.policies.p3 import P3VoI
from askback.runner.episode import REFUSE, _run_episode
from askback.stops.replay import replay
from askback.stops.rules import s1_fires, s2_fires, s2_tau_fires


class TruthSim:
    def __init__(self, domain):
        self.domain = domain

    def answer(self, field, rng):
        return self.domain.truth[field]


def test_s1_fires_on_max_below_threshold() -> None:
    assert not s1_fires({0: 0.5, 1: 0.02}, 0.02)
    assert s1_fires({0: 0.001, 1: 0.0001}, 0.02)
    assert s1_fires({}, 0.02)  # empty U cannot move the pass probability


def test_s2_tau_uses_the_resolvable_share() -> None:
    u = {1, 2, 3, 4, 5}
    assert s2_fires(set())
    assert not s2_fires({3})
    assert s2_tau_fires(set(), u, 0.7)  # 5/5
    assert s2_tau_fires({1}, u, 0.7)  # 4/5
    assert not s2_tau_fires({1, 2}, u, 0.7)  # 3/5
    assert not s2_tau_fires({1}, u, 1.0)
    # V4 algebra, stated literally: |U − G| / max(|U|, 1), never |G| / |U|.
    g, u4 = {1}, {1, 2, 3, 4}
    assert s2_tau_fires(g, u4, 0.7) == (((len(u4) - len(g)) / max(len(u4), 1)) >= 0.7)
    assert s2_tau_fires(g, u4, 0.7) and not (len(g) / len(u4) >= 0.7)  # 0.75 vs 0.25


def _s2_fixture() -> tuple[Domain, object]:
    """Field 0 known ⇒ rules close fields 2–5, so validator_pass is reachable;
    field 1 stays genuinely unresolvable and is the only gap."""
    dom = Domain(
        domains=[[0, 1]] * 6,
        rules=[Rule((0,), (0,), i, 0) for i in (2, 3, 4, 5)],
        truth=[0] * 6,
    )
    return dom, initial_belief(dom).with_value(0, 0)


def test_s2_tau_fires_before_validator_pass_at_t0() -> None:
    """W3 gate: S2(0.7) stops the episode before §9.1 #4 would.

    A12 Behavior B judges t = 0 on the initial post-extraction state:
    U = {1..5}, G = {1}, so |U−G|/|U| = 4/5 = 0.8 ≥ 0.7 fires at T = 0 with no
    turn at all. S0 must execute one turn to reach validator_pass.
    """
    dom, belief = _s2_fixture()
    seed = np.random.SeedSequence(3)

    s0 = _run_episode(
        dom, belief, P0FixedOrder(), TruthSim(dom), DirectSetMerger(), seed,
        k=1, stop_rule="S0",
    )
    s2 = _run_episode(
        dom, belief, P0FixedOrder(), TruthSim(dom), DirectSetMerger(), seed,
        k=1, stop_rule="S2", s2_tau=0.7,
    )

    assert s0.turns == 1 and s0.reason == "validator_pass" and s0.validity
    assert s2.turns == 0 and not s2.records
    assert s2.reason == "rule_stop" and not s2.validity
    assert s2.initial is not None
    assert not s2.initial.validator_pass  # S2 won before the pass existed
    assert s2.initial.u == frozenset({1, 2, 3, 4, 5})
    assert s2.initial.g == frozenset({1})

    # Replay reproduces the zero-turn decision from the initial state alone.
    rep = replay(s2.records, rule="S2", tau=0.7, initial=s2.initial)
    assert (rep.turns, rep.validity, rep.abandoned, rep.reason) == (
        s2.turns, s2.validity, s2.abandoned, s2.reason
    )


def _small(seed: int) -> Domain:
    """5 binary fields, acyclic random rules — small enough that T-09 stays in
    the exact VoI path and the 100-episode sweep stays seconds, not minutes."""
    rng = np.random.default_rng(seed)
    rules = [
        Rule((int(rng.integers(0, out)),), (int(rng.integers(0, 2)),), out,
             int(rng.integers(0, 2)))
        for out in range(1, 5)
        if rng.random() < 0.6
    ]
    truth: list[int] = []
    for f in range(5):
        fired = [
            r.out_val
            for r in rules
            if r.out == f
            and all(truth[p] == v for p, v in zip(r.premises, r.prem_vals))
        ]
        truth.append(fired[0] if fired else int(rng.integers(0, 2)))
    return Domain([[0, 1]] * 5, rules, truth)


def _chain_depth(dom: Domain) -> int:
    memo: dict[int, int] = {}

    def depth(f: int) -> int:
        if f not in memo:
            memo[f] = max(
                (1 + max(depth(p) for p in r.premises)
                 for r in dom.rules if r.out == f),
                default=0,
            )
        return memo[f]

    return max(depth(f) for f in range(dom.n_fields))


def _tier2() -> Domain:
    """5 fields; the binary rule 2 ← (0, 1) is tier 2, so k=2 closes
    0,1 → 2 → 3 → 4 while k=1 cannot start that chain."""
    return Domain(
        domains=[[0, 1]] * 5,
        rules=[
            Rule((0, 1), (0, 0), 2, 0),
            Rule((2,), (0,), 3, 0),
            Rule((3,), (0,), 4, 0),
        ],
        truth=[0] * 5,
    )


def test_t09_family_covers_k1_k2_and_depth() -> None:
    """Family coverage: a k=1-only domain, a k=2-exercising domain, and a
    propagation chain of depth ≥ 2."""
    # k = 1 only: every `_small` rule is unary, so tier 1 is the whole ladder.
    assert all(max((r.tier for r in _small(i).rules), default=1) == 1 for i in range(10))
    # Depth: _small(0) is one hop; _small(1) chains 0 → 1 → 2 → 4.
    assert _chain_depth(_small(0)) == 1
    assert _chain_depth(_small(1)) >= 2

    # k = 2 exercises: same fixture and policy, tier 2 finishes strictly sooner.
    dom = _tier2()
    k1 = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), TruthSim(dom),
        DirectSetMerger(), np.random.SeedSequence(0), k=1,
    )
    k2 = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), TruthSim(dom),
        DirectSetMerger(), np.random.SeedSequence(0), k=2,
    )
    assert k2.turns == 2 and k2.reason == "validator_pass" and k2.validity
    assert k1.turns == 3 and k1.reason == "validator_pass" and k1.validity


@pytest.mark.parametrize(
    "rule,tau,c", [("S0", 1.0, 0.02), ("S1", 1.0, 0.02), ("S1", 1.0, 0.10),
                   ("S2", 0.7, 0.02)]
)
def test_t09_replay_matches_online_on_100_episodes(
    rule: str, tau: float, c: float
) -> None:
    """T-09 as extended by A12/F5: S0, S1(c) and S2(τ) offline == online.

    S1 is included because the record now logs the pre-ask top-of-turn ΔP that
    online S1 consulted, so equality holds by construction, not by luck.
    """
    doms = [_small(i) for i in range(10)] + [_tier2()]
    reasons: set[str] = set()
    for i in range(100):
        dom = doms[i % len(doms)]
        res = _run_episode(
            dom, initial_belief(dom), P3VoI(), TruthSim(dom), DirectSetMerger(),
            np.random.SeedSequence(i),
            k=2, patience=20, stop_rule=rule, s1_c=c, s2_tau=tau,
        )
        rep = replay(res.records, rule=rule, c=c, tau=tau, patience=20,
                     initial=res.initial)
        assert (rep.turns, rep.validity, rep.abandoned, rep.reason) == (
            res.turns, res.validity, res.abandoned, res.reason
        ), (i, rule)
        reasons.add(res.reason)
    # Not vacuous: S2 must actually end some of these episodes, and not all.
    if rule == "S2":
        assert "rule_stop" in reasons and len(reasons) > 1, reasons


def test_replay_is_a_pure_function_of_the_log() -> None:
    dom = _small(4)
    res = _run_episode(
        dom, initial_belief(dom), P3VoI(), TruthSim(dom), DirectSetMerger(),
        np.random.SeedSequence(4), k=2,
    )
    assert replay(res.records, initial=res.initial) == replay(list(res.records), initial=res.initial)


def test_refusal_only_episode_exhausts_patience() -> None:
    dom = _small(1)

    class LazySim:
        def answer(self, field, rng):
            return REFUSE

    res = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), LazySim(), DirectSetMerger(),
        np.random.SeedSequence(1), patience=3,
    )
    assert res.reason == "patience_exhausted" and res.abandoned
    assert replay(res.records, patience=3).abandoned


def test_refused_derivable_passes() -> None:
    """A13: a refused field that forward chaining derives from answered fields
    does not block the validator; the episode still passes.
    """
    dom = Domain(
        domains=[[0, 1]] * 3,
        rules=[Rule((0,), (0,), 1, 0)],
        truth=[0, 0, 0],
    )

    class RefuseOneSim:
        def __init__(self, domain):
            self.domain = domain

        def answer(self, field, rng):
            if field == 1:  # derivable from field 0 via forward chaining
                return REFUSE
            return self.domain.truth[field]

    res = _run_episode(
        dom, initial_belief(dom), P0FixedOrder(), RefuseOneSim(dom),
        DirectSetMerger(), np.random.SeedSequence(0), k=1, patience=5,
    )
    assert res.refusals == 1 and res.refused_count == 1
    assert res.records[1].field == 1
    assert all(r.field != 1 for i, r in enumerate(res.records) if i != 1)
    assert res.reason == "u_empty" and res.passed and res.validity