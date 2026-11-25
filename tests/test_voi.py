"""T-04, T-05: VoI exact/MC agreement and the A4 short-circuit — §7.2, §18."""

from __future__ import annotations

import numpy as np

from askback.belief.state import Belief, initial_belief
from askback.domain.schema import Domain, Rule
from askback.domain.validator import resolvable_set
from askback.engine.sets import unresolved_set
from askback.engine.voi import (
    analytic_delta,
    compute_voi,
    exact_delta,
    exact_ok,
    pass_se,
)


def _random_small(seed: int) -> tuple[Domain, Belief]:
    """3–5 fields, acyclic random rules, hand-spread beliefs (incl. ⊥ mass)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 6))
    rules = []
    for _ in range(int(rng.integers(1, 4))):
        out = int(rng.integers(1, n))
        rules.append(
            Rule(
                (int(rng.integers(0, out)),),
                (int(rng.integers(0, 2)),),
                out,
                int(rng.integers(0, 2)),
            )
        )
    dom = Domain([[0, 1]] * n, rules, [int(rng.integers(0, 2)) for _ in range(n)])
    belief = initial_belief(dom)
    for f in range(n):
        if rng.random() < 0.25:
            p = np.zeros(3)
            p[int(rng.integers(0, 2))] = 1.0
        else:
            p = rng.random(3)
            p /= p.sum()
        belief.probs[f] = p
    return dom, belief


def test_t04a_exact_paths_agree_on_100_fixtures() -> None:
    """Two independent exact routes: per-value conditioning vs the ratio identity."""
    for seed in range(100):
        dom, belief = _random_small(seed)
        for f in unresolved_set(belief, dom):
            assert abs(
                exact_delta(belief, dom, 1, f) - analytic_delta(belief, dom, 1, f)
            ) < 1e-6, (seed, f)


def _mc_fixture(seed: int) -> tuple[Domain, Belief]:
    """9 uncertain binary fields — past A12/R1's 6-field gate, so MC is forced."""
    rng = np.random.default_rng(seed)
    dom = Domain([[0, 1]] * 9, [], [0] * 9)
    belief = initial_belief(dom)
    miss = 0.02 + 0.18 * float(rng.random())
    for f in range(9):
        a = 0.3 + 0.4 * float(rng.random())
        p = np.array([a * (1 - miss), (1 - a) * (1 - miss), miss])
        belief.probs[f] = p / p.sum()
    assert not exact_ok(belief, dom)
    return dom, belief


def test_t04b_mc_agrees_with_exact_within_four_se() -> None:
    for seed in range(100):
        dom, belief = _mc_fixture(seed)
        mc = compute_voi(belief, dom, 1, np.random.default_rng(seed + 1))
        se = pass_se(belief, dom, 1, np.random.default_rng(seed + 2), 2048)
        for f in unresolved_set(belief, dom):
            # A12/Option A: 4·SE, no absolute slack term.
            assert abs(mc[f] - analytic_delta(belief, dom, 1, f)) < 4 * se, (
                seed,
                f,
            )


def test_t04_gate_boundaries() -> None:
    six = Domain([[0, 1]] * 6, [], [0] * 6)
    seven = Domain([[0, 1]] * 7, [], [0] * 7)
    assert exact_ok(initial_belief(six), six)
    assert not exact_ok(initial_belief(seven), seven)
    # ∏|D_f| > 5 000 with ≤6 uncertain fields also drops to MC (A12/R1).
    big = Domain([list(range(6))] * 5, [], [0] * 5)  # 6**5 = 7 776
    assert not exact_ok(initial_belief(big), big)
    mid = Domain([list(range(5))] * 5, [], [0] * 5)  # 5**5 = 3 125
    assert exact_ok(initial_belief(mid), mid)


def test_t05_resolvable_field_has_zero_voi_in_both_modes() -> None:
    # Exact mode.
    dom = Domain(
        domains=[[0, 1], [0, 1], [0, 1]],
        rules=[Rule((0,), (0,), 1, 0), Rule((1,), (0,), 2, 0)],
        truth=[0, 0, 0],
    )
    belief = initial_belief(dom).with_value(0, 0)
    u = unresolved_set(belief, dom)
    r = resolvable_set(belief.probs, dom, 1) & u
    assert r, "fixture must have a resolvable unresolved field"
    voi = compute_voi(belief, dom, 1, np.random.default_rng(0))
    assert all(voi[f] == 0.0 for f in r)

    # MC mode: 9 uncertain fields push past the gate, rule still resolves one.
    dom9 = Domain([[0, 1]] * 10, [Rule((0,), (0,), 9, 0)], [0] * 10)
    b9 = initial_belief(dom9).with_value(0, 0)
    for f in range(1, 10):
        b9.probs[f] = np.array([0.45, 0.45, 0.10])
    assert not exact_ok(b9, dom9)
    u9 = unresolved_set(b9, dom9)
    r9 = resolvable_set(b9.probs, dom9, 1) & u9
    assert r9
    voi9 = compute_voi(b9, dom9, 1, np.random.default_rng(0))
    assert all(voi9[f] == 0.0 for f in r9)


def test_a5_fully_missing_field_uses_uniform_prior() -> None:
    # Single 3-value field, no rules: any answer makes the schema complete.
    dom = Domain([[0, 1, 2]], [], [1])
    belief = initial_belief(dom)
    assert belief.probs[0][-1] == 1.0
    assert abs(exact_delta(belief, dom, 1, 0) - 1.0) < 1e-9
    assert abs(analytic_delta(belief, dom, 1, 0) - 1.0) < 1e-9
    assert abs(compute_voi(belief, dom, 1, np.random.default_rng(0))[0] - 1.0) < 1e-9