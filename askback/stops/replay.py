"""Offline stop replay — PRD §9.3, FR-9; ordering per A11, timing per A12.

Every stop decision is a pure function of the logged :class:`TurnRecord`s, so
replay reproduces the online episode exactly.

Note: online sets ``validity = validator_pass_after[last]`` whichever branch
ends the episode, so replay reads validity from the record and never derives
it. That keeps the `u_empty`-with-a-violation case identical on both paths.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

from askback.runner.episode import InitialState
from askback.stops.rules import s1_threshold, s2_tau_fires, s2_threshold


class ReplayResult(NamedTuple):
    turns: int
    validity: bool
    abandoned: bool
    reason: str


def replay(
    records: Iterable,
    *,
    rule: str = "S0",
    c: float = 0.02,
    tau: float = 1.0,
    patience: int = 20,
    initial: InitialState | None = None,
) -> ReplayResult:
    """Apply §9.1 (A10 order, A12 timing) to a recorded episode.

    ``initial`` is the pre-turn state; A12 Behavior B lets a rule stop end an
    episode at T = 0, so replay cannot reproduce that from records alone.
    """
    turns = 0
    if initial is not None:
        if not initial.u:
            return ReplayResult(0, initial.validator_pass, False, "u_empty")
        if rule == "S1" and s1_threshold(initial.delta_p, c):
            return ReplayResult(0, initial.validator_pass, False, "rule_stop")
        if rule == "S2" and s2_tau_fires(initial.g, initial.u, tau):
            return ReplayResult(0, initial.validator_pass, False, "rule_stop")
        if initial.validator_pass:
            return ReplayResult(0, True, False, "validator_pass")
    for r in records:
        turns += 1
        if turns >= patience:
            return ReplayResult(turns, r.validator_pass_after, True,
                                "patience_exhausted")
        if r.u_empty_after:
            return ReplayResult(turns, r.validator_pass_after, False, "u_empty")
        # A12: the rule reads the current post-state and its logged ΔP.
        if rule == "S1" and s1_threshold(r.delta_p, c):
            return ReplayResult(turns, r.validator_pass_after, False, "rule_stop")
        if rule == "S2" and s2_threshold(
            r.resolvable_count_after, r.u_count_after, tau
        ):
            return ReplayResult(turns, r.validator_pass_after, False, "rule_stop")
        if r.validator_pass_after:
            return ReplayResult(turns, True, False, "validator_pass")
    return ReplayResult(turns, False, True, "truncated")


if __name__ == "__main__":
    from askback.runner.episode import TurnRecord

    def rec(t, dp=0.0, ok=False, u_empty=False, r_count=0, u_count=1):
        return TurnRecord(
            t=t, field=0, delta_p=dp, answer_type="value",
            validator_pass_after=ok, resolved_fraction_after=0.0,
            g_empty_after=False, u_empty_after=u_empty,
            determinism_hash="x", resolvable_count_after=r_count,
            u_count_after=u_count,
        )

    assert replay([rec(0), rec(1, ok=True)]) == ReplayResult(
        2, True, False, "validator_pass"
    )
    assert replay([rec(0), rec(1, ok=True)], rule="S1", c=0.5).reason == "rule_stop"
    assert replay([rec(0, r_count=1), rec(1, ok=True)], rule="S2").reason == "rule_stop"
    assert replay([rec(0, u_empty=True)]) == ReplayResult(1, False, False, "u_empty")
    assert replay([rec(0), rec(1)], patience=2).abandoned
    assert replay([]).reason == "truncated"
    print("replay ok")