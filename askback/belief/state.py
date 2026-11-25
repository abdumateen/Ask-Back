"""Categorical belief state — PRD §6.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from askback.domain.schema import Domain


@dataclass
class Belief:
    """One categorical distribution per field over D_f ∪ {⊥}; ⊥ is the last slot."""

    probs: list[np.ndarray]

    def copy(self) -> "Belief":
        return Belief([p.copy() for p in self.probs])

    def with_extraction(self, f: int, value: int, conf: float) -> "Belief":
        """§6.1: p(v)=c, p(⊥)=1−c, every other value 0."""
        b = self.copy()
        p = np.zeros_like(b.probs[f])
        p[value] = conf
        p[-1] = 1.0 - conf
        b.probs[f] = p
        return b

    def with_missing(self, f: int) -> "Belief":
        """§6.1: p(⊥)=1."""
        b = self.copy()
        p = np.zeros_like(b.probs[f])
        p[-1] = 1.0
        b.probs[f] = p
        return b

    def with_value(self, f: int, value: int) -> "Belief":
        """§6.1: user answer v ⇒ p(v)=1.0."""
        return self.with_extraction(f, value, 1.0)


def initial_belief(domain: Domain) -> Belief:
    """Nothing extracted yet: every field is ⊥ with certainty."""
    return Belief([np.append(np.zeros(len(d)), 1.0) for d in domain.domains])


def from_extraction(
    domain: Domain, extraction: dict[int, tuple[int | None, float]]
) -> Belief:
    """§5: turn an extractor's per-field (value, confidence) map into a belief.

    ``value is None`` means missing and leaves the field at ⊥.
    """
    belief = initial_belief(domain)
    for f, (value, conf) in extraction.items():
        if value is None:
            belief = belief.with_missing(f)
        else:
            belief = belief.with_extraction(f, value, conf)
    return belief


if __name__ == "__main__":
    from askback.domain.generator import generate_domain
    from askback.domain.schema import UNKNOWN, argmax_value, determined

    dom = generate_domain(20250, "low")
    b0 = initial_belief(dom)
    assert all(p[-1] == 1.0 for p in b0.probs)

    b1 = b0.with_extraction(0, 1, 0.7)
    assert np.isclose(b1.probs[0].sum(), 1.0)
    assert np.isclose(b1.probs[0][1], 0.7)
    assert np.isclose(b1.probs[0][UNKNOWN], 0.3)
    assert argmax_value(b1.probs[0]) == 1 and not determined(b1.probs[0])

    b2 = b1.with_value(0, 1)
    assert determined(b2.probs[0]) and argmax_value(b2.probs[0]) == 1
    assert b0.probs[0][UNKNOWN] == 1.0  # source belief untouched

    b3 = b2.with_missing(0)
    assert argmax_value(b3.probs[0]) == UNKNOWN
    print("belief ok")