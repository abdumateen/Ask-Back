"""Gap engine — PRD §6.2, FR-6.

`G = U − R` where `U` is the undetermined set and `R` the set of undetermined
fields the tier-k validator can resolve by forward chaining. Recomputed from
scratch every turn; never cached (FR-6).
"""

from __future__ import annotations

from askback.belief.state import Belief
from askback.domain.schema import Domain, determined
from askback.domain.validator import resolvable_set


def determined_set(belief: Belief, domain: Domain) -> set[int]:
    return {f for f in range(domain.n_fields) if determined(belief.probs[f])}


def unresolved_set(belief: Belief, domain: Domain) -> set[int]:
    return set(range(domain.n_fields)) - determined_set(belief, domain)


def compute_sets(
    belief: Belief, domain: Domain, k: int,
    refused: set[int] | frozenset[int] = set(),
) -> tuple[set[int], set[int], set[int]]:
    """Return `(U, R, G)` for the current turn.

    `refused` fields (PRD A13: answered REFUSE) leave `U` for the rest of the
    episode; they stay in `R` — a refused field may still be derivable by
    forward chaining from fields that were answered.
    """
    U = unresolved_set(belief, domain) - set(refused)
    R = resolvable_set(belief.probs, domain, k)
    return U, R, U - R


if __name__ == "__main__":
    from askback.belief.state import initial_belief
    from askback.domain.generator import generate_domain

    dom = generate_domain(20250, "low")
    bel = initial_belief(dom)

    U, R, G = compute_sets(bel, dom, k=2)
    assert U == set(range(dom.n_fields)), "initial belief has nothing determined"
    assert R == set(), "R needs determined premises, which the initial belief lacks"
    assert G == U

    truth_belief = bel
    for f in range(dom.n_fields):
        truth_belief = truth_belief.with_value(f, dom.truth[f])
    U, R, G = compute_sets(truth_belief, dom, k=2)
    assert not U and not R and not G, "fully determined truth belief has no gap"

    print("sets ok", len(U), len(R), len(G))