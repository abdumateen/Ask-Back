"""Rule-specific stops — PRD §9.2 as replaced by A10.

S0 never fires early. S1(c) fires when no unresolved field moves the pass
probability. S2(τ) fires when at least a τ share of the unresolved fields is
resolvable — `u - g == R ∩ U`, so the gap set is all S2 needs.
"""

from __future__ import annotations

from askback.extract.interface import Field


def s1_threshold(delta_max: float, c: float) -> bool:
    return delta_max < c


def s1_fires(delta_p: dict[Field, float], c: float) -> bool:
    """§9.2 S1(c): `max_{f∈U} ΔP_f < c`."""
    return s1_threshold(max(delta_p.values(), default=0.0), c)


def s2_threshold(resolvable: int, unresolved: int, tau: float) -> bool:
    """§9.2 S2(τ) as a share, so replay can use the logged counts."""
    return resolvable / max(unresolved, 1) >= tau


def s2_tau_fires(g: set[Field], u: set[Field], tau: float) -> bool:
    """A10: `|R ∩ U| / max(|U|,1) >= τ`, with `R ∩ U = u - g`."""
    return s2_threshold(len(u - g), len(u), tau)


def s2_fires(g: set[Field]) -> bool:
    """§15.1 locked τ = 1.0 specialisation: `R ∩ U == ∅`."""
    return not g


if __name__ == "__main__":
    assert not s1_fires({0: 0.5}, 0.02)
    assert s1_fires({0: 0.01, 1: 0.001}, 0.02)
    assert s1_fires({}, 0.02)
    assert s2_fires(set())
    assert not s2_fires({3})
    assert s2_tau_fires(set(), {1, 2, 3, 4, 5}, 0.7)
    assert s2_tau_fires({1}, {1, 2, 3, 4, 5}, 0.7)  # 4/5
    assert not s2_tau_fires({1, 2}, {1, 2, 3, 4, 5}, 0.7)  # 3/5
    assert s2_threshold(3, 5, 0.6) and not s2_threshold(3, 5, 0.7)
    print("rules ok")