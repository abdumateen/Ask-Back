"""T-03: resolvable_set vs exhaustive reachable-fact closure — PRD §18."""

from __future__ import annotations

import numpy as np
import pytest

from askback.domain.generator import generate_domain
from askback.domain.schema import Domain
from askback.domain.validator import resolvable_set


def brute_resolvable(known: dict[int, int], domain: Domain, k: int) -> set[int]:
    """Naive rescan-to-fixpoint closure; independent of the product's bookkeeping."""
    facts = dict(known)
    added: set[int] = set()
    while True:
        grew = False
        for r in domain.rules:
            if r.tier > k or r.out in facts:
                continue
            if all(facts.get(p) == v for p, v in zip(r.premises, r.prem_vals)):
                facts[r.out] = r.out_val
                added.add(r.out)
                grew = True
        if not grew:
            return added


def belief_of(known: dict[int, int], domain: Domain) -> list[np.ndarray]:
    out = []
    for f, values in enumerate(domain.domains):
        p = np.zeros(len(values) + 1)
        if f in known:
            p[known[f]] = 1.0
        else:
            p[-1] = 1.0
        out.append(p)
    return out


@pytest.mark.parametrize("group", range(40))
def test_resolvable_matches_brute_force(group: int) -> None:
    domain = generate_domain(group, "low" if group % 2 == 0 else "high")
    rng = np.random.default_rng(1000 + group)

    for _ in range(5):
        n_known = int(rng.integers(9, 12))
        chosen = rng.choice(domain.n_fields, size=n_known, replace=False)
        known = {int(f): domain.truth[int(f)] for f in chosen}

        for k in (1, 2):
            got = resolvable_set(belief_of(known, domain), domain, k)
            want = brute_resolvable(known, domain, k)
            assert got == want, f"seed={group} k={k} known={sorted(known)}"