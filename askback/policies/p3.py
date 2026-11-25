"""P3 — VoI greedy: highest ΔP in `U` (PRD §8).

Ties: a zero-VoI field loses to any positive one; remaining ties by ascending
field index. Pure-zero `U` therefore falls back to P0 order. No answerability
weight, no cost term, no lookahead.
"""

from __future__ import annotations

import numpy as np

from askback.belief.state import Belief
from askback.extract.interface import Field


class P3VoI:
    def select(
        self,
        u: set[Field],
        delta_p: dict[Field, float],
        belief: Belief,
        rng: np.random.Generator,
    ) -> Field:
        return max(u, key=lambda f: (delta_p.get(f, 0.0), -f))


if __name__ == "__main__":
    from askback.belief.state import initial_belief
    from askback.domain.schema import Domain

    dom = Domain(domains=[[0, 1]] * 4, rules=[], truth=[0] * 4)
    bel = initial_belief(dom)
    p = P3VoI()
    rng = np.random.default_rng(0)
    assert p.select({0, 1, 2, 3}, {0: 0.0, 1: 0.3, 2: 0.1, 3: 0.3}, bel, rng) == 1
    assert p.select({2, 3}, {2: 0.0, 3: 0.0}, bel, rng) == 2  # P0 fallback
    assert p.select({3}, {}, bel, rng) == 3
    print("p3 ok")