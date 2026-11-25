"""Policy contract — PRD §15.1 (signature locked), §8.

The policies may not use answerability weights, question cost, or lookahead.
They choose only from `U`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from askback.belief.state import Belief
from askback.extract.interface import Field


class Policy(Protocol):
    def select(
        self,
        u: set[Field],
        delta_p: dict[Field, float],
        belief: Belief,
        rng: np.random.Generator,
    ) -> Field: ...