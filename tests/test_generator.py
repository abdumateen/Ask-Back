"""T-01: generator acyclic, conflict-free, deterministic — PRD §18."""

from __future__ import annotations

import pytest

from askback.domain.generator import generate_domain, premises_compatible
from askback.domain.schema import Rule


@pytest.mark.parametrize("seed", range(1000))
def test_generator_invariants(seed: int) -> None:
    density = "low" if seed % 2 == 0 else "high"
    dom = generate_domain(seed, density)

    assert dom.n_fields == 15
    assert all(2 <= len(d) <= 5 for d in dom.domains)

    for r in dom.rules:
        assert max(r.premises) < r.out  # acyclic
        assert len(r.premises) == len(r.prem_vals)
        assert all(0 <= v < len(dom.domains[p]) for p, v in zip(r.premises, r.prem_vals))
        assert 0 <= r.out_val < len(dom.domains[r.out])

    assert len(dom.rules) == len(set(dom.rules))  # deduplicated

    for i, r1 in enumerate(dom.rules):
        for r2 in dom.rules[i + 1 :]:
            if r1.out == r2.out and r1.out_val != r2.out_val:
                assert not premises_compatible(r1, r2), "§4.2 invariant violated"

    for f in range(dom.n_fields):
        fired = [
            r.out_val
            for r in dom.rules
            if r.out == f
            and all(dom.truth[p] == v for p, v in zip(r.premises, r.prem_vals))
        ]
        assert len(set(fired)) <= 1
        assert dom.truth[f] == (fired[0] if fired else dom.truth[f])
        assert 0 <= dom.truth[f] < len(dom.domains[f])


@pytest.mark.parametrize("density", ["low", "high"])
def test_generator_deterministic(density: str) -> None:
    for seed in range(50):
        a = generate_domain(seed, density)
        b = generate_domain(seed, density)
        assert a.domains == b.domains
        assert a.rules == b.rules
        assert a.truth == b.truth


TARGET_RULES = {"low": 6, "high": 20}


def test_rule_count_shortfall() -> None:
    """FR-4.3 permits fewer than R rules; record how often that happens (A8)."""
    stats = {d: [0, 0] for d in TARGET_RULES}  # [shortfalls, total]
    worst: list[tuple[int, str, int]] = []

    for density, target in TARGET_RULES.items():
        for seed in range(1000):
            dom = generate_domain(seed, density)
            assert len(dom.rules) <= target
            stats[density][1] += 1
            if len(dom.rules) < target:
                stats[density][0] += 1
                worst.append((len(dom.rules), density, seed))

    worst.sort()
    for density, (short, total) in stats.items():
        print(
            f"\nshortfall {density}: {short}/{total} "
            f"({100.0 * short / total:.1f}%) target={TARGET_RULES[density]}"
        )
    print(f"smallest accepted sets: {worst[:5]}")


def test_premises_compatible() -> None:
    a = Rule((0, 1), (0, 0), 3, 1)
    b = Rule((0,), (0,), 4, 1)  # shares field 0 with the same value
    c = Rule((1, 2), (1, 2), 3, 0)  # shares field 1 with a different value
    assert premises_compatible(a, b)
    assert not premises_compatible(a, c)