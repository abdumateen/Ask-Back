"""Analytic user simulator — PRD §10.2, §10.4.

`answer` is §10.2 verbatim except for the added `domain` parameter: the
distractor draw needs `D_f`, which §10.2's signature omits (recorded in A10).
"""

from __future__ import annotations

import numpy as np

from askback.domain.schema import Domain
from askback.seeds import knowledge_rng
from askback.sim.personas import Persona

REFUSE = "REFUSE"


def answer(
    persona: Persona,
    domain: Domain,
    field: int,
    truth: list[int],
    knowledge_set: set[int],
    rng: np.random.Generator,
) -> int | str:
    if field not in knowledge_set:
        return REFUSE
    if rng.random() < persona.noise:
        others = [v for v in range(len(domain.domains[field])) if v != truth[field]]
        return int(rng.choice(others))
    return truth[field]


def sample_knowledge_set(domain: Domain, rho: float, seed: np.random.SeedSequence) -> set[int]:
    """`K ⊆ F`, each field included independently at rate `rho` (§10.2)."""
    rng = np.random.default_rng(knowledge_rng(seed))
    return {f for f in range(domain.n_fields) if rng.random() < rho}


class AnalyticAnswerer:
    """Binds persona, domain, truth and knowledge set so the runner needs only
    `answer(field, rng)`."""

    def __init__(self, persona: Persona, domain: Domain, knowledge_set: set[int]) -> None:
        self.persona = persona
        self.domain = domain
        self.truth = domain.truth
        self.knowledge_set = knowledge_set

    def answer(self, field: int, rng: np.random.Generator) -> int | str:
        return answer(
            self.persona, self.domain, field, self.truth, self.knowledge_set, rng
        )


if __name__ == "__main__":
    from askback.domain.generator import generate_domain
    from askback.sim.personas import PERSONAS

    dom = generate_domain(20250, "high")
    seed = np.random.SeedSequence(1)

    coop = AnalyticAnswerer(PERSONAS["cooperative"], dom, set(range(dom.n_fields)))
    for f in range(dom.n_fields):
        got = coop.answer(f, np.random.default_rng(f))
        assert got == dom.truth[f] and got != REFUSE

    ignorant = AnalyticAnswerer(PERSONAS["cooperative"], dom, set())
    assert all(
        ignorant.answer(f, np.random.default_rng(f)) == REFUSE for f in range(dom.n_fields)
    )

    noisy = AnalyticAnswerer(PERSONAS["knowledgeable_noisy"], dom, set(range(dom.n_fields)))
    for f in range(dom.n_fields):
        got = noisy.answer(f, np.random.default_rng(f))
        assert got != REFUSE and got in range(len(dom.domains[f])), got

    assert sample_knowledge_set(dom, 1.0, seed) == set(range(dom.n_fields))
    assert sample_knowledge_set(dom, 0.0, seed) == set()
    assert sample_knowledge_set(dom, 0.7, seed) == sample_knowledge_set(dom, 0.7, seed)
    print("answerer ok")