"""Extractor and merger contracts — PRD §5."""

from __future__ import annotations

from typing import Protocol, TypeAlias

from askback.belief.state import Belief

Field: TypeAlias = int
Value: TypeAlias = int


class Extractor(Protocol):
    def extract(self, nl_text: str) -> dict[Field, tuple[Value | None, float]]:
        """Return (value, calibrated_confidence) or (None, 0.0) per field."""


class AnswerMerger(Protocol):
    def merge(self, belief: Belief, field: Field, value: Value) -> tuple[Belief, bool]:
        """Return (new_belief, conflict)."""