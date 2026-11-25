"""Gap engine — PRD §6.2, FR-6: `G = U − R`, recomputed per turn."""

from __future__ import annotations

import pytest

from askback.belief.state import initial_belief
from askback.domain.generator import generate_domain
from askback.domain.schema import determined
from askback.engine.sets import compute_sets, determined_set, unresolved_set


@pytest.mark.parametrize("density", ["low", "high"])
def test_initial_belief_has_no_resolvable_set(density: str) -> None:
    dom = generate_domain(20250, density)
    belief = initial_belief(dom)

    U, R, G = compute_sets(belief, dom, k=2)
    assert U == set(range(dom.n_fields))
    assert R == set()
    assert G == U


@pytest.mark.parametrize("seed", range(20))
def test_fully_determined_belief_has_empty_sets(seed: int) -> None:
    dom = generate_domain(seed, "high")
    belief = initial_belief(dom)
    for f in range(dom.n_fields):
        belief = belief.with_value(f, dom.truth[f])

    U, R, G = compute_sets(belief, dom, k=2)
    assert not U and not R and not G
    assert determined_set(belief, dom) == set(range(dom.n_fields))
    assert unresolved_set(belief, dom) == set()
    assert all(determined(belief.probs[f]) for f in range(dom.n_fields))


def test_r_is_subset_of_u_and_g_is_the_remainder() -> None:
    dom = generate_domain(20250, "high")
    belief = initial_belief(dom)
    # Determine only field 0; whatever chaining resolves is R.
    belief = belief.with_value(0, dom.truth[0])

    U, R, G = compute_sets(belief, dom, k=2)
    assert R <= U
    assert G == U - R
    assert not (R & G)


def test_sets_are_recomputed_not_cached() -> None:
    dom = generate_domain(20250, "low")
    belief = initial_belief(dom)
    U1, _, _ = compute_sets(belief, dom, k=2)

    belief = belief.with_value(0, dom.truth[0])
    U2, _, _ = compute_sets(belief, dom, k=2)
    assert len(U2) < len(U1), "a new determined field must shrink U"