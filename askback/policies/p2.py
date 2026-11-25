"""P2 — greedy uncertainty: `argmin_f max_v p_f(v)`, ties by lowest index (PRD §8)."""

from __future__ import annotations

import numpy as np

from askback.belief.state import Belief
from askback.extract.interface import Field


class P2Uncertainty:
    def select(
        self,
        u: set[Field],
        delta_p: dict[Field, float],
        belief: Belief,
        rng: np.random.Generator,
    ) -> Field:
        # §8: `argmin_f max_v p_f(v)` over D_f. The ⊥ column is not a value, so a
        # field with p_missing = 0.20 reads as 0.8, not 1.0 (V1, A12).
        return min(u, key=lambda f: (float(belief.probs[f][:-1].max()), f))


if __name__ == "__main__":
    import numpy as np

    from askback.belief.state import initial_belief
    from askback.domain.schema import Domain

    dom = Domain(domains=[[0, 1]] * 6, rules=[], truth=[0] * 6)
    bel = initial_belief(dom)
    bel.probs[0] = np.array([0.90, 0.10, 0.0])
    bel.probs[1] = np.array([0.90, 0.10, 0.0])
    bel.probs[2] = np.array([0.50, 0.50, 0.0])
    bel.probs[3] = np.array([0.90, 0.10, 0.0])
    bel.probs[4] = np.array([0.60, 0.40, 0.0])
    bel.probs[5] = np.array([0.50, 0.50, 0.0])

    p = P2Uncertainty()
    assert p.select({0, 1, 2, 3, 4, 5}, {}, bel, np.random.default_rng(0)) == 2
    assert p.select({4, 5}, {}, bel, np.random.default_rng(0)) == 5
    assert p.select({0, 4}, {}, bel, np.random.default_rng(0)) == 4
    # V1: a fully-missing field scores 0.0 (no value has mass), not 1.0.
    bel.probs[5] = np.array([0.0, 0.0, 1.0])
    assert p.select({0, 5}, {}, bel, np.random.default_rng(0)) == 5
    print("p2 ok")