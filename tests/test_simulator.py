"""T-07: simulator refusal and distractor behaviour — PRD §10.2, §10.4."""

from __future__ import annotations

import numpy as np

from askback.domain.generator import generate_domain
from askback.seeds import MASTER_SEED, episode_seed
from askback.sim.answerer import REFUSE, AnalyticAnswerer, answer, sample_knowledge_set
from askback.sim.personas import PERSONAS


def _dom():
    return generate_domain(20250, "high")


def test_answers_truth_when_cooperative() -> None:
    dom = _dom()
    sim = AnalyticAnswerer(PERSONAS["cooperative"], dom, set(range(dom.n_fields)))
    for f in range(dom.n_fields):
        assert sim.answer(f, np.random.default_rng(f)) == dom.truth[f]


def test_refuses_when_field_outside_knowledge_set() -> None:
    dom = _dom()
    sim = AnalyticAnswerer(PERSONAS["cooperative"], dom, set())
    for f in range(dom.n_fields):
        assert sim.answer(f, np.random.default_rng(f)) == REFUSE


def test_noisy_persona_gives_valid_values_never_refuses() -> None:
    dom = _dom()
    sim = AnalyticAnswerer(
        PERSONAS["knowledgeable_noisy"], dom, set(range(dom.n_fields))
    )
    for f in range(dom.n_fields):
        for s in range(20):
            got = sim.answer(f, np.random.default_rng(s))
            assert got != REFUSE
            assert got in range(len(dom.domains[f])), (f, got)
    # noise=0.15: at least one distractor over 15 fields x 20 seeds
    assert any(
        sim.answer(f, np.random.default_rng(s)) != dom.truth[f]
        for f in range(dom.n_fields)
        for s in range(20)
    )


def test_zero_noise_never_distracts() -> None:
    dom = _dom()
    for name in ("cooperative", "impatient_expert"):
        sim = AnalyticAnswerer(PERSONAS[name], dom, set(range(dom.n_fields)))
        assert all(
            sim.answer(f, np.random.default_rng(s)) == dom.truth[f]
            for f in range(dom.n_fields)
            for s in range(10)
        )


def test_knowledge_set_is_deterministic_and_hits_rate() -> None:
    dom = _dom()
    seed = episode_seed(MASTER_SEED, 0, 0, 1, 0)

    assert sample_knowledge_set(dom, 1.0, seed) == set(range(dom.n_fields))
    assert sample_knowledge_set(dom, 0.0, seed) == set()
    assert sample_knowledge_set(dom, 0.7, seed) == sample_knowledge_set(dom, 0.7, seed)


def test_module_answer_needs_domain_for_distractor_choice() -> None:
    """A10: §10.2's signature omits D_f, so the distractor draw takes `domain`."""
    dom = _dom()
    got = answer(
        PERSONAS["knowledgeable_noisy"], dom, 0, dom.truth, set(range(dom.n_fields)),
        np.random.default_rng(0),
    )
    assert got != REFUSE and 0 <= got < len(dom.domains[0])