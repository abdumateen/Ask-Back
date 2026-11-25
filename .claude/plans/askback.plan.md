# Plan: AskBack — Milestone W1

**Source PRD**: `C:\Users\Abdul Mateen\Downloads\Plan-Doc.md` (v1.0) **as amended by** `PRD-v1.1-amendments.md` (A1–A7, ratified 2026-11-22)
**Selected Milestone**: W1 — domain + validator + stub extractor + episode runner skeleton (T-01, T-02, T-03)
**Complexity**: Medium

## Summary

Build the deterministic simulation core: the D1 rule-graph generator (`generate_domain`), the tier-k validator and its resolvable-set closure, the stub extractor and direct-set merger, a minimal belief state, and the episode state machine skeleton. Everything downstream (VoI, policies, replay, experiments) depends on these types and on the byte-level determinism guaranteed here. No policies, no stop rules, no VoI in W1.

## Patterns to Mirror

No patterns to mirror. `D:\AskBack` is empty (only `.` and `..`), so there is no existing naming convention, error style, logging format, or test layout to follow. Conventions are therefore taken from the PRD itself:

| Category | Source | Pattern |
|---|---|---|
| Naming | PRD §15 (module tree) | `askback/<layer>/<module>.py`; layers are `domain`, `extract`, `belief`, `engine`, `policies`, `stops`, `sim`, `runner`, `eval` |
| Signatures | PRD §15.1 | Public signatures are fixed verbatim; do not rename, do not add required params |
| Determinism | PRD §16 | All randomness from `np.random.SeedSequence` spawn keys; never `np.random.seed`, never global RNG |
| Config | PRD §14.1 | YAML config keys as written; `configs/core.yaml` is the single source of run parameters |
| Tests | PRD §18 | Test IDs `T-NN` map 1:1 to test functions; fixtures are hand-built, no external data files |

## Files to Change

| File | Action | Why |
|---|---|---|
| `pyproject.toml` | CREATE | Package metadata, `pytest` config, deps: `numpy`, `pandas`, `matplotlib`, `pytest`, `pyyaml` (PRD §15 as amended by A3) |
| `askback/__init__.py` | CREATE | Package marker |
| `askback/domain/schema.py` | CREATE | `Field`, `Value`, `Rule`, `Domain`, `UNKNOWN` (`⊥`), `determined()` |
| `askback/domain/generator.py` | CREATE | `generate_domain(seed, density)` per PRD §4.3 |
| `askback/domain/validator.py` | CREATE | `validate(belief, domain, k)` §4.4 and `resolvable_set(belief, domain, k)` §4.5 |
| `askback/belief/state.py` | CREATE | `Belief` categorical state + §6.1 update rules |
| `askback/belief/merge.py` | CREATE | `DirectSetMerger` per §5 |
| `askback/extract/interface.py` | CREATE | `Extractor` and `AnswerMerger` protocols per §5 |
| `askback/extract/stub.py` | CREATE | `StubExtractor` with `p_clean .50 / p_uncertain .30 / p_missing .20` |
| `askback/extract/p2_adapter.py` | CREATE | `P2Extractor` raising `NotImplementedError` (§5) |
| `askback/runner/episode.py` | CREATE | State machine skeleton per §11; loop body, termination asserts (FR-11) |
| `tests/test_generator.py` | CREATE | T-01 |
| `tests/test_validator.py` | CREATE | T-02 |
| `tests/test_resolvable.py` | CREATE | T-03, with brute-force reference implementation |
| `prereg/README.md` | CREATE | PRD §17 template, committed before any experiment (§19.6) |

## Tasks

### Task 1: Scaffold package and pin determinism helpers
- **Action**: Create `pyproject.toml` (requires-python `>=3.11`, deps exactly `numpy`, `pandas`, `matplotlib`, `pytest`, `pyyaml` per A3), the `askback/` package tree from §15, and a `seeds.py` helper implementing §16 spawn-key composition: `episode_seed(master, policy_id, persona_id, density_id, episode_idx)` and `turn_rng(episode_seed, t)`, `voi_rng(episode_seed, t)`.
- **Mirror**: PRD §16 seed table; PRD §15 module tree.
- **Validate**: `python -c "from askback.seeds import episode_seed; print(episode_seed(20250,0,0,1,0))"` prints a value and prints it identically on a second run.

### Task 2: Domain schema types
- **Action**: `schema.py` with `UNKNOWN = -1` sentinel, `Rule(premises, prem_vals, out, out_val)` where `tier_requirement(rule) = len(premises)`, `Domain(domains, rules, truth)`, and `determined(belief_f) -> bool` implementing FR-2.1: `argmax(p) != UNKNOWN and max(p) >= 0.99` (§2 as amended by A7).
- **Mirror**: PRD §4.2 rule form; §2 determined-field definition.
- **Validate**: Constructing a `Rule` with `max(premises) >= out` raises `ValueError` (acyclicity guard, §4.2).

### Task 3: D1 generator
- **Action**: Implement A1 verbatim — module-level `premises_compatible(r1, r2)`; rule loop with `seen`-set dedupe on `(premises, prem_vals, out)`; reject any new rule whose output matches an existing rule with a different `out_val` and compatible premises; then forward-sample `truth` in ascending field index, asserting at most one firing rule per field.
- **Mirror**: `PRD-v1.1-amendments.md` A1 code block, verbatim. Do not re-derive.
- **Validate**: T-01 — 1000 seeds × both densities: no exception, no rule violation in `truth`, §4.2 invariant holds on every generated rule set, identical serialized output on rerun.

### Task 4: Tier-k validator
- **Action**: `validate(belief, domain, k)` per §4.4: seed `vals` from determined fields, propagate only rules with `len(premises) <= k` to a fixpoint (bounded by `F` passes), fail on determined conflict, fail with `"undetermined"` if any field unresolved, then re-check tier-≤k rules and return `(True, "pass")`. Higher-tier rules are never applied nor checked (R5).
- **Mirror**: PRD §4.4 code shape, including the `(bool, reason)` return.
- **Validate**: T-02 — 20 hand-built fixtures covering pass, undetermined, violation, tier-1 rule ignored under `k=1` when its premises are undetermined, tier-2 rule ignored under `k=1` but applied under `k=2`, and a propagation chain of depth 3.

### Task 5: Resolvable-set closure
- **Action**: Implement A2 verbatim — least-fixpoint forward chaining over tier-≤k rules; `R` holds only fields not determined at entry; return `set()` on violation. Include `brute_force_resolvable()` in the test module as the independent reference: enumerate consistent assignments in §4.3 forward-sampling order and check the forced value is unique per field.
- **Mirror**: PRD §4.5 comment ("iterative closure"); PRD §6.2 usage — never cached across turns (FR-6).
- **Validate**: T-03 — 200 random beliefs × `k ∈ {1,2}`: closure equals brute force exactly.

### Task 6: Belief state and merger
- **Action**: `Belief` holding one categorical array per field over `D_f ∪ {⊥}`; constructors for the §6.1 events (extraction with confidence, extraction missing, user answer, refusal). `DirectSetMerger.merge` sets `p(value)=1.0` and all others 0, returning `(belief, conflict)` where `conflict` is true iff a previously determined value differed.
- **Mirror**: PRD §5 merger contract; PRD §6.1 update table.
- **Validate**: Round-trip test — merging the same value twice yields `conflict=False` on the second call; merging a different determined value yields `conflict=True`.

### Task 7: Extractor layer
- **Action**: `interface.py` protocols per §5. `StubExtractor` draws per field independently from the categorical `p_clean / p_uncertain / p_missing`, taking its RNG as a parameter. `P2Extractor` raises `NotImplementedError("P2 artifacts unavailable")` on construction. The runner refuses `extractor: p2`.
- **Mirror**: PRD §5 stub table and protocol signatures.
- **Validate**: 10,000-draw test — empirical frequencies within ±0.02 of configured probabilities; uncertain values always lie in `D_f`; missing always yields `(None, 0.0)`.

### Task 8: Episode state machine skeleton
- **Action**: `run_episode(cfg, domain, policy, persona, seed) -> Trajectory` implementing §11 turn order: recompute `determined/U/R/G` → validator check → `U == ∅` check → patience check → placeholder `delta_p` (all zeros, W1 only) → policy select → simulator answer → merge → log record. Emit the §9.3 turn record fields now so replay has a stable log schema. Assert `t <= min(patience, F + 1)` every iteration (FR-11).
- **Mirror**: PRD §11 state-machine block; §9.3 log record field list.
- **Validate**: Skeleton run with `P0` and persona `realistic` completes 10 seeds without assertion failure and produces well-formed turn records.

### Task 9: Pre-registration file
- **Action**: Create `prereg/README.md` containing the §17 template verbatim plus a committed timestamp line; git-commit it before `runs/` exists (§19.6).
- **Mirror**: PRD §17 block.
- **Validate**: File exists, contains all seven §17 lines, and its commit precedes any `runs/` path.

## Validation

```bash
python -m pytest tests -q
python -m pytest tests -q --tb=short -k "determinism"
python -c "from askback.domain.generator import generate_domain as g; a=g(20250,'high'); b=g(20250,'high'); assert a == b"
```

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| **R1 — PRD §4.3 ground truth is wrong.** The listed code draws a random value for every field and merely `break`s out of the rule loop on conflict, then rejects and resamples the whole truth. Nothing ever *applies* a rule consequence, so accepted truths are those where no antecedent matches — rule edges go inert and the mechanism claim (C1) collapses. | High | Forward-sample rule consequences in ascending index order (Task 3). Requires a written PRD amendment under the change-control clause. |
| **R2 — `resolvable_set` is a comment, not an algorithm** (§4.5). T-03's brute force is the only real spec. | High | Treat brute force as the contract (Task 5); write the closure to match it. |
| **R3 — No plotting dependency.** §15 allows `numpy, pandas, pytest, pyyaml` only; §19 requires three PNGs and §13.3 needs plots. `pandas` cannot emit PNG. | High | Needs a PRD amendment: add `matplotlib`, or cut deliverables 2–4. Blocks W5/W6, not W1. |
| **R4 — Exact-mode VoI skips the R short-circuit.** §7.2 returns from `exact_ok` before the `f in R` branch, so exact mode can report `ΔP_f != 0` for resolvable fields, violating T-05. | Medium | Apply the short-circuit inside the exact branch too. W3 concern; recorded now. |
| **R5 — Division by zero in §7.2.** `pv / (1 - belief[f][UNKNOWN])` is undefined when `p(⊥) = 1`, and fully-missing fields are in `U`. | Medium | Guard: if `p(⊥) == 1`, return `ΔP_f = 0.0`. W3 concern; recorded now. |
| **R6 — T-04 states two tolerances.** "`|Δ| < 1e-6`... within MC SE ×3" mixes an exact-vs-MC bound with a statistical bound. | Medium | Split into two assertions: exact vs analytic `<= 1e-6`, MC vs exact within `3 SE`. Confirm intent before W3. |
| **R7 — Generator emits duplicate rules.** §4.2 forbids identical `(premises, values)` pairs; §4.3 never deduplicates. With 20 rules over 2–5-value domains, collisions are possible. | Low | Deduplicate on insert (Task 3). |
| ~~R1, R2, R3, R7~~ | — | **Closed** by `PRD-v1.1-amendments.md` A1, A2, A3 — the mitigation text in those rows is that amendment's spec; implement it verbatim. |
| **Budget.** T-12 caps A1 at 10 min on 8 cores; VoI is `O(|U| · max|D| · 2048)` batch validations per turn over 400 episodes. | Medium | Measure single-episode wall time in W3; reduce `n_mc` only via PRD amendment. |

## Acceptance

- [ ] All tasks complete
- [ ] `python -m pytest tests -q` green with T-01, T-02, T-03 implemented
- [ ] Generator and validator reproduce byte-identically on rerun for seed 20250
- [ ] `prereg/README.md` committed before `runs/` exists
- [ ] Patterns mirrored from the PRD, not reinvented
- [ ] No policies, stop rules, or VoI code present (W2/W3 scope)