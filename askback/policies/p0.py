"""P0 — fixed order: lowest field index in `U` (PRD §8)."""

from __future__ import annotations

import numpy as np

from askback.belief.state import Belief
from askback.extract.interface import Field


class P0FixedOrder:
    def select(
        self,
        u: set[Field],
        delta_p: dict[Field, float],
        belief: Belief,
        rng: np.random.Generator,
    ) -> Field:
        return min(u)


if __name__ == "__main__":
    from askback.belief.state import initial_belief
    from askback.domain.generator import generate_domain

    dom = generate_domain(20250, "high")
    bel = initial_belief(dom)
    p = P0FixedOrder()
    U = {3, 7, 11}
    assert p.select(U, {f: 0.0 for f in U}, bel, np.random.default_rng(0)) == 3
    assert p.select({0}, {}, bel, np.random.default_rng(0)) == 0
    print("p0 ok")