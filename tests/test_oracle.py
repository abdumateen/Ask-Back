"""§12 bounds: the oracle answers with the minimum subset size."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from askback.belief.state import from_extraction
from askback.domain.generator import generate_domain
from askback.domain.schema import determined
from askback.domain.validator import validate
from askback.extract.stub import StubExtractor
from askback.runner.oracle import episode_bounds, no_interaction, oracle_questions
from askback.sim.answerer import sample_knowledge_set

DOMAIN = generate_domain(20250, "high")
EXTRACTION = StubExtractor(DOMAIN, np.random.default_rng(3)).extract("")
KNOWLEDGE = sample_knowledge_set(DOMAIN, 0.7, np.random.SeedSequence(5))


def _brute_force(extraction: dict, knowledge: set[int], k: int = 1) -> int | None:
    base = from_extraction(DOMAIN, extraction)
    askable = sorted(f for f in knowledge if not determined(base.probs[f]))
    for size in range(0, len(askable) + 1):
        for subset in combinations(askable, size):
            belief = base
            for f in subset:
                belief = belief.with_value(f, DOMAIN.truth[f])
            if validate(belief.probs, DOMAIN, k)[0]:
                return size
    return None


def test_oracle_matches_exhaustive_search_on_small_k() -> None:
    small = sorted(KNOWLEDGE)[:6]
    knowledge = set(small)
    assert oracle_questions(DOMAIN, EXTRACTION, knowledge) == _brute_force(
        EXTRACTION, knowledge
    )


def test_bounds_are_deterministic() -> None:
    first = episode_bounds(DOMAIN, EXTRACTION, KNOWLEDGE)
    assert first == episode_bounds(DOMAIN, EXTRACTION, KNOWLEDGE)
    assert set(first) == {"no_interaction", "oracle_questions"}
    q = first["oracle_questions"]
    assert q is None or 0 <= q <= DOMAIN.n_fields


def test_no_interaction_is_the_zero_question_case() -> None:
    passed = no_interaction(DOMAIN, EXTRACTION)
    q = oracle_questions(DOMAIN, EXTRACTION, KNOWLEDGE)
    assert not passed or q == 0
    assert passed == (q == 0)  # zero questions means nothing was needed