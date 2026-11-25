"""Persona table — PRD §10.1. Frozen values; the PRD fixes all four rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    rho: float  # P(field in knowledge set)
    noise: float  # P(deliberate distractor | knows the field)
    patience: int  # turns before `patience_exhausted`


PERSONAS: dict[str, Persona] = {
    "cooperative": Persona("cooperative", rho=1.00, noise=0.00, patience=106),
    "realistic": Persona("realistic", rho=0.70, noise=0.10, patience=8),
    "impatient_expert": Persona("impatient_expert", rho=1.00, noise=0.00, patience=4),
    "knowledgeable_noisy": Persona("knowledgeable_noisy", rho=1.00, noise=0.15, patience=8),
}


if __name__ == "__main__":
    assert len(PERSONAS) == 4
    assert PERSONAS["impatient_expert"].patience == 4
    assert all(0.0 <= p.rho <= 1.0 and 0.0 <= p.noise <= 1.0 for p in PERSONAS.values())
    print("personas ok")