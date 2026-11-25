"""§12 per-episode bounds: no-interaction and oracle.

Both answer the same question from opposite ends: how many questions does a
cell need? The no-interaction bound asks nothing and reads turn-0 validity; the
oracle searches ``knowledge_set`` by ascending subset size and returns the size
of the first subset whose true values make the tier-k validator pass.
"""

from __future__ import annotations

from itertools import combinations

from askback.belief.state import from_extraction
from askback.domain.schema import Domain, determined
from askback.domain.validator import validate


def no_interaction(domain: Domain, extraction: dict, k: int = 1) -> bool:
    """Turn-0 validity: does the extraction alone pass the tier-k validator?"""
    return validate(from_extraction(domain, extraction).probs, domain, k)[0]


def oracle_questions(
    domain: Domain, extraction: dict, knowledge_set: set[int], k: int = 1
) -> int | None:
    """§12 oracle: minimum questions to pass, or ``None`` when unreachable.

    Only fields the extraction left undetermined can help, so the search runs
    over those. ``2^15`` worst case (PRD §12). The first success at the smallest
    size is the minimum: the validator only ever needs *more* determined fields,
    and truth values never contradict the schema.
    """
    base = from_extraction(domain, extraction)
    if validate(base.probs, domain, k)[0]:
        return 0
    askable = sorted(f for f in knowledge_set if not determined(base.probs[f]))
    for size in range(1, len(askable) + 1):
        for subset in combinations(askable, size):
            belief = base
            for f in subset:
                belief = belief.with_value(f, domain.truth[f])
            if validate(belief.probs, domain, k)[0]:
                return size
    return None


def episode_bounds(
    domain: Domain, extraction: dict, knowledge_set: set[int], k: int = 1
) -> dict:
    """§12's two reference values for one episode."""
    return {
        "no_interaction": no_interaction(domain, extraction, k),
        "oracle_questions": oracle_questions(domain, extraction, knowledge_set, k),
    }


if __name__ == "__main__":
    import numpy as np

    from askback.domain.generator import generate_domain
    from askback.extract.stub import StubExtractor
    from askback.sim.answerer import sample_knowledge_set

    dom = generate_domain(20250, "high")
    ext = StubExtractor(dom, np.random.default_rng(3)).extract("")
    knowledge = sample_knowledge_set(dom, 0.7, np.random.SeedSequence(5))

    bounds = episode_bounds(dom, ext, knowledge)
    assert bounds == episode_bounds(dom, ext, knowledge), "not deterministic"
    q = bounds["oracle_questions"]
    assert q is None or 0 <= q <= dom.n_fields
    assert not bounds["no_interaction"] or q == 0
    print("oracle ok", bounds)