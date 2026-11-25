"""D1 rule-graph generator — PRD §4.3 as amended by A1."""

from __future__ import annotations

import numpy as np

from askback.domain.schema import Domain, Rule


def premises_compatible(r1: Rule, r2: Rule) -> bool:
    """True when no shared premise field is assigned different values."""
    for p1, v1 in zip(r1.premises, r1.prem_vals):
        for p2, v2 in zip(r2.premises, r2.prem_vals):
            if p1 == p2 and v1 != v2:
                return False
    return True


def generate_domain(seed: int, density: str) -> Domain:
    rng = np.random.default_rng(seed)
    n = 15
    sizes = rng.choice([2, 3, 4, 5], size=n)
    domains = [list(range(int(s))) for s in sizes]

    R = 6 if density == "low" else 20
    p_single = 0.8 if density == "low" else 0.3

    rules: list[Rule] = []
    by_out: dict[int, list[Rule]] = {}
    seen: set[tuple[tuple[int, ...], tuple[int, ...], int]] = set()

    # FR-4.3 allows an accepted rule set smaller than R when draws are rejected,
    # so the draw loop is bounded instead of unbounded.
    attempts = 0
    max_attempts = 200 * R
    while len(rules) < R and attempts < max_attempts:
        attempts += 1
        out = int(rng.integers(1, n))
        m = 1 if rng.random() < p_single else 2
        m = min(m, out)
        premises = tuple(
            sorted(int(x) for x in rng.choice(out, size=m, replace=False))
        )
        prem_vals = tuple(int(rng.integers(len(domains[p]))) for p in premises)
        out_val = int(rng.integers(len(domains[out])))

        sig = (premises, prem_vals, out)
        if sig in seen:  # A1 / R7 dedupe
            continue

        r_new = Rule(premises, prem_vals, out, out_val)
        if any(
            r2.out_val != out_val and premises_compatible(r_new, r2)
            for r2 in by_out.get(out, ())
        ):  # A1 / R1 consistency
            continue

        rules.append(r_new)
        by_out.setdefault(out, []).append(r_new)
        seen.add(sig)

    # Forward-sample truth in ascending field index; every premise index is
    # lower than its rule's output index, so premises are already fixed.
    truth: list[int] = [0] * n
    for f in range(n):
        fired = [
            r.out_val
            for r in rules
            if r.out == f
            and all(truth[p] == v for p, v in zip(r.premises, r.prem_vals))
        ]
        assert len(set(fired)) <= 1, "§4.2 invariant violated"
        truth[f] = fired[0] if fired else int(rng.integers(len(domains[f])))

    return Domain(domains, rules, truth)


if __name__ == "__main__":
    a = generate_domain(20250, "high")
    b = generate_domain(20250, "high")
    assert a.domains == b.domains and a.rules == b.rules and a.truth == b.truth
    for r in a.rules:
        assert max(r.premises) < r.out
        assert all(v >= 0 for v in r.prem_vals)
    for r in a.rules:
        if all(a.truth[p] == v for p, v in zip(r.premises, r.prem_vals)):
            assert a.truth[r.out] == r.out_val, "truth violates its own rule"
    print("generator ok", len(a.rules), "rules")