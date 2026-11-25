"""Answer mergers — PRD §5 adapter contract."""

from __future__ import annotations

from askback.belief.state import Belief
from askback.domain.schema import argmax_value, determined


class DirectSetMerger:
    """Trust the answer absolutely: p(value)=1.0, everything else 0."""

    def merge(self, belief: Belief, field: int, value: int) -> tuple[Belief, bool]:
        """Return the updated belief and whether a known value was contradicted."""
        f = belief.probs[field]
        prior = argmax_value(f) if determined(f) else None
        conflict = prior is not None and prior != value
        return belief.with_value(field, value), conflict


if __name__ == "__main__":
    from askback.belief.state import initial_belief
    from askback.domain.generator import generate_domain

    dom = generate_domain(20250, "low")
    m = DirectSetMerger()

    b, conflict = m.merge(initial_belief(dom), 0, 1)
    assert not conflict and b.probs[0][1] == 1.0 and b.probs[0].sum() == 1.0

    b2, conflict2 = m.merge(b, 0, 1)
    assert not conflict2  # same value again: no conflict

    _, conflict3 = m.merge(b, 0, 2)
    assert conflict3  # determined value differed
    print("merge ok")