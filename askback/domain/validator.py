"""Tier-k validator and resolvable set — PRD §4.4, §4.5 as amended by A2."""

from __future__ import annotations

import numpy as np

from askback.domain.schema import Domain, argmax_value, determined

#: Reason codes returned by :func:`validate`.
VIOLATION = "violation"
UNDETERMINED = "undetermined"
PASS = "pass"


def _known_values(
    belief: list[np.ndarray], domain: Domain
) -> dict[int, int | None]:
    return {
        f: (argmax_value(belief[f]) if determined(belief[f]) else None)
        for f in range(domain.n_fields)
    }


def validate(belief: list[np.ndarray], domain: Domain, k: int) -> tuple[bool, str]:
    """Apply only tier-≤k rules to a fixpoint, then check completeness.

    Returns ``(True, "pass")``, ``(False, "violation")`` or
    ``(False, "undetermined")``. Higher-tier rules are neither applied nor
    checked (PRD R5).
    """
    vals = _known_values(belief, domain)

    for _ in range(domain.n_fields + 1):
        changed = False
        for r in domain.rules:
            if r.tier > k:
                continue
            if not all(
                vals[p] is not None and vals[p] == v
                for p, v in zip(r.premises, r.prem_vals)
            ):
                continue
            if vals[r.out] is None:
                vals[r.out] = r.out_val
                changed = True
            elif vals[r.out] != r.out_val:
                return False, VIOLATION
        if not changed:
            break

    if any(v is None for v in vals.values()):
        return False, UNDETERMINED

    for r in domain.rules:
        if r.tier > k:
            continue
        if all(vals[p] == v for p, v in zip(r.premises, r.prem_vals)):
            if vals[r.out] != r.out_val:
                return False, VIOLATION
    return True, PASS


def resolvable_set(
    belief: list[np.ndarray], domain: Domain, k: int
) -> set[int]:
    """Fields whose value is forced to a unique value by tier-k chaining.

    Recomputed every turn (FR-4.5); never cached across turns. Returns an empty
    set on violation — the caller terminates the episode.
    """
    vals = _known_values(belief, domain)
    determined_initial = {f for f, v in vals.items() if v is not None}
    R: set[int] = set()
    changed = True
    while changed:
        changed = False
        for r in domain.rules:
            if r.tier > k:  # tier restriction (PRD R5)
                continue
            if vals[r.out] is not None:
                continue
            if all(
                vals[p] is not None and vals[p] == v
                for p, v in zip(r.premises, r.prem_vals)
            ):
                vals[r.out] = r.out_val
                if r.out not in determined_initial:
                    R.add(r.out)
                changed = True
    return R


if __name__ == "__main__":
    from askback.domain.generator import generate_domain

    dom = generate_domain(20250, "high")

    bel = [np.append(np.zeros(len(d)), 1.0) for d in dom.domains]
    assert validate(bel, dom, 1) == (False, UNDETERMINED)
    assert resolvable_set(bel, dom, 1) == set()

    truth_bel = []
    for f, d in enumerate(dom.domains):
        p = np.zeros(len(d) + 1)
        p[dom.truth[f]] = 1.0
        truth_bel.append(p)
    assert validate(truth_bel, dom, 1) == (True, PASS)
    assert resolvable_set(truth_bel, dom, 1) == set()
    print("validator ok", len(dom.rules), "rules")