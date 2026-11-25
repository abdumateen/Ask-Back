"""T-06: P0/P1/P2 determinism, tie-breaking, membership — PRD §8."""

from __future__ import annotations

import numpy as np
import pytest

from askback.belief.state import initial_belief
from askback.domain.generator import generate_domain
from askback.domain.schema import Domain
from askback.policies.p0 import P0FixedOrder
from askback.policies.p1 import P1Random
from askback.policies.p2 import P2Uncertainty
from askback.policies.p3 import P3VoI


def _domain(n: int = 6) -> Domain:
    return Domain(domains=[[0, 1]] * n, rules=[], truth=[0] * n)


@pytest.mark.parametrize("policy_cls", [P0FixedOrder, P1Random, P2Uncertainty, P3VoI])
def test_selects_from_u_and_repeats_under_one_seed(policy_cls) -> None:
    dom = generate_domain(3, "high")
    belief = initial_belief(dom)
    U = set(range(dom.n_fields))

    a = policy_cls().select(U, {f: 0.0 for f in U}, belief, np.random.default_rng(11))
    b = policy_cls().select(U, {f: 0.0 for f in U}, belief, np.random.default_rng(11))
    assert a == b
    assert a in U


def test_p0_picks_lowest_index() -> None:
    belief = initial_belief(_domain())
    assert P0FixedOrder().select({3, 7, 11}, {}, belief, np.random.default_rng(0)) == 3


def test_p1_spreads_but_is_seed_reproducible() -> None:
    dom = generate_domain(20250, "high")
    belief = initial_belief(dom)
    U = set(range(dom.n_fields))
    policy = P1Random()

    picks = {policy.select(U, {}, belief, np.random.default_rng(s)) for s in range(100)}
    assert picks <= U
    assert len(picks) > 1


def test_p2_lowest_confidence_then_lowest_index() -> None:
    belief = initial_belief(_domain())
    belief.probs[0] = np.array([0.90, 0.10, 0.0])
    belief.probs[1] = np.array([0.90, 0.10, 0.0])
    belief.probs[2] = np.array([0.50, 0.50, 0.0])
    belief.probs[3] = np.array([0.90, 0.10, 0.0])
    belief.probs[4] = np.array([0.60, 0.40, 0.0])
    belief.probs[5] = np.array([0.50, 0.50, 0.0])

    policy = P2Uncertainty()
    rng = np.random.default_rng(0)
    assert policy.select({0, 1, 2, 3, 4, 5}, {}, belief, rng) == 2  # 2 ties with 5
    assert policy.select({4, 5}, {}, belief, rng) == 5  # max 0.6 vs 0.5
    assert policy.select({0, 4}, {}, belief, rng) == 4
    # V1: ⊥ mass is not evidence. Field 0's true max is 0.30, field 1's 0.35;
    # reading the ⊥ column would score field 0 at 0.40 and pick it for the
    # wrong reason.
    belief.probs[0] = np.array([0.30, 0.30, 0.40])
    belief.probs[1] = np.array([0.35, 0.35, 0.30])
    assert policy.select({0, 1}, {}, belief, rng) == 0


def test_p3_top_voi_then_p0_order() -> None:
    belief = initial_belief(_domain())
    policy = P3VoI()
    rng = np.random.default_rng(0)

    # §8: zero-VoI loses to any positive one; remaining ties by lowest index.
    assert policy.select({0, 1, 2, 3}, {0: 0.0, 1: 0.2, 2: 0.2, 3: 0.1}, belief, rng) == 1
    assert policy.select({2, 4, 5}, {2: 0.0, 4: 0.0, 5: 0.0}, belief, rng) == 2
    # Missing entries count as zero VoI, not as an error.
    assert policy.select({1, 2}, {2: 0.3}, belief, rng) == 2