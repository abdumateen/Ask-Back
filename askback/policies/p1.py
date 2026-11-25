"""P1 — uniform random over `U`, drawn from the turn RNG (PRD §8)."""

from __future__ import annotations

import numpy as np

from askback.belief.state import Belief
from askback.extract.interface import Field


class P1Random:
    def select(
        self,
        u: set[Field],
        delta_p: dict[Field, float],
        belief: Belief,
        rng: np.random.Generator,
    ) -> Field:
        return sorted(u)[int(rng.integers(len(u)))]


if __name__ == "__main__":
    from askback.belief.state import initial_belief
    from askback.domain.generator import generate_domain

    dom = generate_domain(20250, "high")
    bel = initial_belief(dom)
    U = set(range(dom.n_fields))
    p = P1Random()

    picks = {p.select(U, {}, bel, np.random.default_rng(s)) for s in range(100)}
    assert picks <= U and len(picks) > 1, "must spread over U"
    same = [p.select(U, {}, bel, np.random.default_rng(7)) for _ in range(3)]
    assert len(set(same)) == 1, "same seed must reproduce the same pick"
    print("p1 ok", len(picks))