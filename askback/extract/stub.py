"""Simulated extraction — PRD §5 StubExtractor."""

from __future__ import annotations

import numpy as np

from askback.domain.schema import Domain
from askback.extract.interface import Field, Value

P_CLEAN = 0.50
P_UNCERTAIN = 0.30
P_MISSING = 0.20
CONF_UNCERTAIN = 0.30
BLIND_SPOT_THRESHOLD = 0.5


class StubExtractor:
    """Per-field categorical draw: clean (conf 1.0), uncertain (conf 0.30), missing."""

    def __init__(self, domain: Domain, rng: np.random.Generator) -> None:
        self.domain = domain
        self.rng = rng

    def extract(self, nl_text: str = "") -> dict[Field, tuple[Value | None, float]]:
        out: dict[Field, tuple[Value | None, float]] = {}
        for f, values in enumerate(self.domain.domains):
            u = float(self.rng.random())
            if u < P_CLEAN:
                out[f] = (self.domain.truth[f], 1.0)
            elif u < P_CLEAN + P_UNCERTAIN:
                out[f] = (int(self.rng.integers(len(values))), CONF_UNCERTAIN)
            else:
                out[f] = (None, 0.0)
        return out


def blind_spot(confidence: float) -> bool:
    """§5: a field is a blind spot when its extraction confidence is below 0.5."""
    return confidence < BLIND_SPOT_THRESHOLD


if __name__ == "__main__":
    from askback.domain.generator import generate_domain

    dom = generate_domain(20250, "high")
    a = StubExtractor(dom, np.random.default_rng(7)).extract("")
    b = StubExtractor(dom, np.random.default_rng(7)).extract("")
    assert a == b and set(a) == set(range(dom.n_fields))
    for f, (v, c) in a.items():
        assert (v is None) == (c == 0.0)
        assert v is None or 0 <= v < len(dom.domains[f])
        assert blind_spot(c) == (c < 0.5)
    assert {c for _, c in a.values()} <= {1.0, CONF_UNCERTAIN, 0.0}
    print("stub ok")