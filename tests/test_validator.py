"""T-02: validator passes iff all determined and tier-≤k rules hold — PRD §18."""

from __future__ import annotations

import numpy as np
import pytest

from askback.domain.schema import Domain, Rule
from askback.domain.validator import PASS, UNDETERMINED, VIOLATION, validate

# Four binary fields. tier-1: 0=0→1=0, 0=1→1=1, 1=1→2=0, 2=0→3=0.
# tier-2: 0=0 AND 1=1 → 3=1 (unreachable under the tier-1 rules).
DOMAIN = Domain(
    domains=[[0, 1]] * 4,
    rules=[
        Rule((0,), (0,), 1, 0),
        Rule((0,), (1,), 1, 1),
        Rule((1,), (1,), 2, 0),
        Rule((0, 1), (0, 1), 3, 1),
        Rule((2,), (0,), 3, 0),
    ],
)


def belief_of(known: dict[int, int]) -> list[np.ndarray]:
    out = []
    for f in range(DOMAIN.n_fields):
        p = np.zeros(3)  # slots 0, 1, then ⊥
        if f in known:
            p[known[f]] = 1.0
        else:
            p[-1] = 1.0
        out.append(p)
    return out


CASES = [
    ({}, 1, UNDETERMINED),
    ({}, 2, UNDETERMINED),
    ({0: 0}, 1, UNDETERMINED),
    ({0: 0}, 2, UNDETERMINED),
    ({0: 0, 1: 0, 2: 0, 3: 0}, 1, PASS),
    ({0: 0, 1: 0, 2: 0, 3: 1}, 1, VIOLATION),
    ({0: 1, 1: 1, 2: 0, 3: 0}, 1, PASS),
    ({0: 1, 1: 0, 2: 1, 3: 1}, 1, VIOLATION),
    ({0: 1, 1: 1, 2: 1, 3: 1}, 1, VIOLATION),
    ({0: 0, 1: 0, 2: 1, 3: 1}, 1, PASS),
    ({0: 0, 1: 0, 2: 1, 3: 1}, 2, PASS),
    ({0: 0, 1: 1, 2: 0, 3: 1}, 2, VIOLATION),
    ({0: 0, 1: 1, 2: 0, 3: 1}, 1, VIOLATION),
    ({0: 1, 1: 1, 2: 0}, 1, PASS),
    ({0: 0, 1: 0}, 1, UNDETERMINED),
    ({0: 0, 1: 0, 2: 0}, 1, PASS),
    ({0: 1, 1: 1}, 1, PASS),
    ({0: 1, 1: 1}, 2, PASS),
    ({2: 0}, 1, UNDETERMINED),
    ({0: 1, 3: 1}, 1, VIOLATION),
]


@pytest.mark.parametrize("known,k,expected", CASES)
def test_validate(known: dict[int, int], k: int, expected: str) -> None:
    ok, reason = validate(belief_of(known), DOMAIN, k)
    assert reason == expected
    assert ok == (expected == PASS)


def test_case_count() -> None:
    assert len(CASES) == 20