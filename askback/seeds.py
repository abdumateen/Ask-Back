"""Seed composition per PRD §16.

Every random draw in the project comes from a SeedSequence whose spawn key is
composed here. Never call ``np.random.seed``; never touch the global RNG.
"""

from __future__ import annotations

import numpy as np

MASTER_SEED = 20250


def _compose(master: int, key: tuple[int, ...]) -> np.random.SeedSequence:
    return np.random.SeedSequence(int(master), spawn_key=tuple(int(k) for k in key))


def episode_seed(
    master: int,
    policy_id: int,
    persona_id: int,
    density_id: int,
    episode_idx: int,
) -> np.random.SeedSequence:
    """§16: episode seed from (policy_id, persona_id, density_id, episode_idx)."""
    return _compose(master, (policy_id, persona_id, density_id, episode_idx))


def turn_rng(seed: np.random.SeedSequence, turn_idx: int) -> np.random.Generator:
    """§16: turn RNG, spawn key (turn_idx,)."""
    return np.random.default_rng(
        np.random.SeedSequence(
            entropy=seed.entropy, spawn_key=tuple(seed.spawn_key) + (int(turn_idx),)
        )
    )


def voi_rng(seed: np.random.SeedSequence, turn_idx: int) -> np.random.Generator:
    """§16: VoI simulation RNG, spawn key (turn_idx, 1)."""
    return np.random.default_rng(
        np.random.SeedSequence(
            entropy=seed.entropy, spawn_key=tuple(seed.spawn_key) + (int(turn_idx), 1)
        )
    )


def sim_rng(seed: np.random.SeedSequence, turn_idx: int) -> np.random.Generator:
    """Simulator answer RNG, spawn key (turn_idx, 2).

    Separate from the policy stream (turn_idx,) so P1's draw and the answerer's
    noise/refusal draw cannot shift each other.
    """
    return np.random.default_rng(
        np.random.SeedSequence(
            entropy=seed.entropy, spawn_key=tuple(seed.spawn_key) + (int(turn_idx), 2)
        )
    )


# Episode-level key. Per-turn keys are (t,), (t, 1), (t, 2); this 2-tuple
# collides with none of them.
_KNOWLEDGE_KEY = (0, 9)


def knowledge_rng(seed: np.random.SeedSequence) -> np.random.Generator:
    """RNG for the persona's knowledge set (§10.2), drawn once per episode."""
    return np.random.default_rng(
        np.random.SeedSequence(
            entropy=seed.entropy, spawn_key=tuple(seed.spawn_key) + _KNOWLEDGE_KEY
        )
    )


if __name__ == "__main__":
    a = episode_seed(MASTER_SEED, 0, 0, 1, 0)
    b = episode_seed(MASTER_SEED, 0, 0, 1, 0)
    assert a.spawn_key == b.spawn_key
    assert turn_rng(a, 0).integers(1 << 30) == turn_rng(b, 0).integers(1 << 30)
    assert voi_rng(a, 0).integers(1 << 30) != turn_rng(a, 0).integers(1 << 30)
    print("seeds ok", a.generate_state(1).tolist())