"""Core domain types — PRD §4.2, §2 (as amended by A7)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Sentinel for "value not known". Belief vectors store its probability in the
#: last slot, so ``belief[f][UNKNOWN]`` is ``belief[f][-1]``.
UNKNOWN = -1

#: FR-2.1: a field counts as determined at or above this posterior mass.
DETERMINED_THRESHOLD = 0.99


@dataclass(frozen=True)
class Rule:
    """IF premises = prem_vals THEN out = out_val, with all premises < out."""

    premises: tuple[int, ...]
    prem_vals: tuple[int, ...]
    out: int
    out_val: int

    def __post_init__(self) -> None:
        if not self.premises:
            raise ValueError("rule needs at least one premise")
        if len(self.premises) != len(self.prem_vals):
            raise ValueError("premises and prem_vals length mismatch")
        highest = max(self.premises)
        if highest >= self.out:
            raise ValueError(
                f"rule is not acyclic: premise field {highest} >= out field {self.out}"
            )

    @property
    def tier(self) -> int:
        """Tier requirement = number of premises (PRD §4.2)."""
        return len(self.premises)


@dataclass
class Domain:
    domains: list[list[int]]
    rules: list[Rule] = field(default_factory=list)
    truth: list[int] = field(default_factory=list)

    @property
    def n_fields(self) -> int:
        return len(self.domains)


def argmax_value(belief_f: np.ndarray) -> int:
    """Most likely value of one field, or ``UNKNOWN`` if ⊥ is the argmax."""
    p = np.asarray(belief_f, dtype=float)
    top = int(p.argmax())
    return UNKNOWN if top == p.size - 1 else top


def determined(belief_f: np.ndarray) -> bool:
    """FR-2.1: argmax is a real value AND its mass is at least 0.99."""
    p = np.asarray(belief_f, dtype=float)
    top = int(p.argmax())
    return top != p.size - 1 and float(p[top]) >= DETERMINED_THRESHOLD


if __name__ == "__main__":
    r = Rule((0,), (1,), 3, 2)
    assert r.tier == 1
    try:
        Rule((4,), (0,), 3, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("acyclicity guard did not fire")

    known = np.array([0.0, 1.0, 0.0])  # slots 0,1 are values; slot 2 is ⊥
    assert determined(known) and argmax_value(known) == 1
    assert not determined(np.array([0.0, 0.0, 1.0]))  # ⊥ holds the mass
    assert argmax_value(np.array([0.0, 0.0, 1.0])) == UNKNOWN
    soft = np.array([0.5, 0.49, 0.01])
    assert not determined(soft) and argmax_value(soft) == 0
    print("schema ok")