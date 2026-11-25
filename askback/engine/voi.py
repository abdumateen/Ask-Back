"""VoI estimator — PRD §7.2, FR-7; A4 exact short-circuit, A5 uniform prior.

`ΔP_f = E_{v∼p_f(·|¬⊥)}[P(pass | f=v)] − P(pass)`.

Exact when the joint is small enough (§7.2 gate), Monte Carlo with common
random numbers otherwise. One `rng` per turn (§7.2, A11): the same `S` serves
every field and every value.
"""

from __future__ import annotations

import numpy as np

from askback.belief.state import Belief
from askback.domain.schema import Domain, argmax_value, determined
from askback.domain.validator import resolvable_set
from askback.engine.sets import unresolved_set

N_MC = 2048
EXACT_MAX_UNCERTAIN = 6
EXACT_MAX_JOINT = 5_000
_EPS = 1e-12


def exact_ok(belief: Belief, domain: Domain) -> bool:
    """§7.2 gate, tightened by A12/R1: ≤6 uncertain fields, ∏|D_f| ≤ 5 000."""
    total = 1
    n_uncertain = 0
    for f in range(domain.n_fields):
        if determined(belief.probs[f]):
            continue
        n_uncertain += 1
        total *= len(domain.domains[f])
    return n_uncertain <= EXACT_MAX_UNCERTAIN and total <= EXACT_MAX_JOINT


def validate_batch(samples: np.ndarray, domain: Domain, k: int) -> np.ndarray:
    """§4.4 validator over a batch of hypothetical snapshots.

    Rows are int arrays, `-1` for ⊥. Forward chaining to fixpoint, then a
    conflict sweep, then completeness: a row passes iff no determined value
    contradicts a tier-≤k rule and every field ends up known.
    """
    vals = np.array(samples, dtype=np.int64, copy=True)
    n = vals.shape[0]
    ok = np.ones(n, dtype=bool)
    rules = [r for r in domain.rules if r.tier <= k]

    for _ in range(domain.n_fields + 1):
        changed = False
        for r in rules:
            prem = np.ones(n, dtype=bool)
            for p, v in zip(r.premises, r.prem_vals):
                prem &= vals[:, p] == v
            unknown = vals[:, r.out] == -1
            fire = prem & unknown & ok
            if fire.any():
                vals[fire, r.out] = r.out_val
                changed = True
            ok &= ~(prem & ~unknown & (vals[:, r.out] != r.out_val))
        if not changed:
            break

    return ok & (vals != -1).all(axis=1)


def sample_joint(
    belief: Belief, rng: np.random.Generator, n_mc: int
) -> np.ndarray:
    """§7.2: draw `n_mc` joint snapshots; ⊥ is encoded `-1`."""
    out = np.empty((n_mc, len(belief.probs)), dtype=np.int64)
    for f, p in enumerate(belief.probs):
        slot = rng.choice(len(p), size=n_mc, p=p)
        out[:, f] = np.where(slot == len(p) - 1, -1, slot)
    return out


def _joint_mass(
    belief: Belief,
    domain: Domain,
    k: int,
    fields: list[int],
    *,
    pin: tuple[int, int] | None = None,
    drop_unknown: int | None = None,
) -> float:
    """Σ P(s)·1[pass(s)] over `fields`, with determined fields fixed at argmax.

    `pin=(f, v)` overrides field `f`; `f` must not be in `fields`.
    `drop_unknown=f` zeroes the mass where `f` stays ⊥ (used by A4's identity).
    """
    if fields:
        grids = [np.arange(len(domain.domains[f]) + 1) for f in fields]
        slots = np.stack(
            [m.ravel() for m in np.meshgrid(*grids, indexing="ij")], axis=1
        )
    else:
        slots = np.zeros((1, 0), dtype=np.int64)

    states = np.full((slots.shape[0], domain.n_fields), -1, dtype=np.int64)
    for f in range(domain.n_fields):
        if determined(belief.probs[f]):
            states[:, f] = argmax_value(belief.probs[f])

    probs = np.ones(slots.shape[0])
    for j, f in enumerate(fields):
        n = len(domain.domains[f])
        col = slots[:, j]
        states[:, f] = np.where(col == n, -1, col)
        probs *= belief.probs[f][col]

    if pin is not None:
        states[:, pin[0]] = pin[1]
    if drop_unknown is not None:
        probs = np.where(states[:, drop_unknown] == -1, 0.0, probs)

    mask = validate_batch(states, domain, k)
    return float(probs @ mask)


def exact_delta(belief: Belief, domain: Domain, k: int, f: int) -> float:
    """Reference ΔP: condition on `f` value by value, re-enumerate each time.

    A5: a fully-missing field uses a uniform prior over `D_f` (denominator 1).
    """
    u = sorted(unresolved_set(belief, domain))
    others = [g for g in u if g != f]
    base = _joint_mass(belief, domain, k, u)

    n = len(domain.domains[f])
    p = belief.probs[f]
    denom = 1.0 - p[-1]
    weights = np.full(n, 1.0 / n) if denom <= _EPS else p[:n] / denom

    acc = 0.0
    for v in range(n):
        if weights[v] == 0.0:
            continue
        acc += float(weights[v]) * _joint_mass(
            belief, domain, k, others, pin=(f, v)
        )
    return acc - base


def analytic_delta(belief: Belief, domain: Domain, k: int, f: int) -> float:
    """Closed form: `(Σ_{s_f≠⊥} p(s)·pass(s)) / (1 − p_f(⊥)) − P(pass)`.

    Same quantity as :func:`exact_delta` by algebra, not by construction, so
    T-04a can compare the two paths. Falls back to the reference for the
    fully-missing uniform case, where the denominator vanishes.
    """
    u = sorted(unresolved_set(belief, domain))
    base = _joint_mass(belief, domain, k, u)
    denom = 1.0 - belief.probs[f][-1]
    if denom <= _EPS:
        return exact_delta(belief, domain, k, f)
    num = _joint_mass(belief, domain, k, u, drop_unknown=f)
    return num / denom - base


def pass_se(
    belief: Belief, domain: Domain, k: int, rng: np.random.Generator, n_mc: int
) -> float:
    """A6: `SE_MC = std(pass_indicator over S) / sqrt(n_mc)`."""
    mask = validate_batch(sample_joint(belief, rng, n_mc), domain, k)
    return float(mask.std() / np.sqrt(n_mc))


def compute_voi(
    belief: Belief,
    domain: Domain,
    k: int,
    rng: np.random.Generator,
    n_mc: int = N_MC,
) -> dict[int, float]:
    """§15.1: ΔP for every unresolved field. `rng` is unused in exact mode."""
    u = sorted(unresolved_set(belief, domain))
    if not u:
        return {}
    resolvable = resolvable_set(belief.probs, domain, k)  # A4

    if exact_ok(belief, domain):
        return {
            f: (0.0 if f in resolvable else analytic_delta(belief, domain, k, f))
            for f in u
        }

    samples = sample_joint(belief, rng, n_mc)
    base = float(validate_batch(samples, domain, k).mean())
    out: dict[int, float] = {}
    for f in u:
        if f in resolvable:
            out[f] = 0.0
            continue
        n = len(domain.domains[f])
        p = belief.probs[f]
        denom = 1.0 - p[-1]
        weights = np.full(n, 1.0 / n) if denom <= _EPS else p[:n] / denom
        acc = 0.0
        for v in range(n):
            w = float(weights[v])
            if w == 0.0:
                continue
            pin = samples.copy()
            pin[:, f] = v
            acc += w * float(validate_batch(pin, domain, k).mean())
        out[f] = acc - base
    return out


if __name__ == "__main__":
    from askback.belief.state import initial_belief
    from askback.domain.schema import Rule

    # field 0 = 0 ⇒ field 1 = 0. Both ⊥: asking 0 buys a coin flip, asking 1
    # buys nothing (0 stays unknown either way).
    dom = Domain(
        domains=[[0, 1], [0, 1]], rules=[Rule([0], [0], 1, 0)], truth=[0, 0]
    )
    bel = initial_belief(dom)
    rng = np.random.default_rng(0)
    voi = compute_voi(bel, dom, 1, rng, 512)
    assert abs(voi[0] - 0.5) < 1e-9, voi
    assert abs(voi[1] - 0.0) < 1e-9, voi

    # A4: once 0 is determined the pair resolves, so both are free.
    det = bel.with_value(0, 0)
    assert compute_voi(det, dom, 1, rng, 512) == {1: 0.0}

    # A5 + both paths agree on the fully-missing single field.
    one = Domain(domains=[[0, 1, 2]], rules=[], truth=[1])
    b1 = initial_belief(one)
    assert abs(exact_delta(b1, one, 1, 0) - 1.0) < 1e-9
    assert abs(analytic_delta(b1, one, 1, 0) - 1.0) < 1e-9
    print("voi ok")