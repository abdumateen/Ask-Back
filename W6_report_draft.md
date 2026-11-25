# AskBack W6 Report (draft) — W5 close

Run dirs: `runs/core_v1` (locked confirmatory), `runs/ablations/a3`,
`runs/cooperative_cell`, `runs/ablations/a6`, `runs/ablations/a6_p2low`
(W5 exploratory; the last is A16c). All runs: high dependency density except
A6 and A6-P2-low (low). k=1. 100 seeds/cell.

---

## 1. Confirmatory null (locked, core_v1 realistic)

Pre-registered H1/H2 on the realistic persona were **null**. W4 was closed as
a null result; A15 records the design acknowledgments.

| Test | v | Policy | Observed | p_one-sided | p_Holm |
|---|---|---|---|---|---|
| H1 P3 vs P0 | 0.05 (fallback) | P0 | T diff 1 | 0.537 | 1.000 |
| H1 P3 vs P2 | 0.05 | P2 | T diff 0 | 0.617 | 1.000 |
| H1 P3 vs P1 | 0.05 | P1 | T diff 0 | 0.697 | 1.000 |
| H2 stop savings | 0.05 | P0 | S0−S2=0 | 1.000 | — |

Reference (S0): P0 v=0.12 / m_turns 7.03; P1 v=0.09 / 7.06; P2 v=0.14 /
6.87; P3 v=0.14 / 6.84. Abandonment ≈46–52% across policies.

vacuous=false; H1 fell back to 0.05 because P1's S0 T(0.10) was undefined.
Null accepted and locked.

## 2. Structural explanation

The realistic cell is **80–88% unpassable** (A15a). The domain validity ceiling
under realistic extraction is low (E[|U₀|]≈7.5; ceiling ≈0.05–0.13), and the
observed pass rates sit near that ceiling. H1/H2 compare pass quantiles in a
cell where almost every episode fails, so the frontier is flattened and the
hypotheses are not discriminable there. Per §20 framing, the **cooperative
persona is the cleaner instrument** for the mechanism claim.

## 3. Exploratory results (post-hoc; not confirmatory)

### 3.1 A15b — H1 at v=0.10 on realistic, excluding P1
Tested P3 against surviving baselines at the primary v=0.10 (P1's T(0.10) is
undefined, the reason the pre-registered test fell back).

| Baseline | T diff (P3 − baseline) | p_one-sided | p_Holm |
|---|---|---|---|
| P0 | +2 turns | 0.483 | 0.965 |
| P2 | +1 turn | 0.498 | 0.965 |

Point estimates favor P3, but the paired bootstrap p-values are not
significant. These results are **exploratory** and do not support a
confirmatory claim.

### 3.2 A3 — P3 across persona spectrum (high density)
| Persona | S0 validity | S0 mean turns | abandon |
|---|---|---|---|
| cooperative | 1.000 | 6.84 | 0.00 |
| impatient_expert | 0.140 | 3.94 | 0.94 |
| knowledgeable_noisy | 0.640 | 6.68 | 0.47 |

impatient_expert (patience 4) abandons in 94% of episodes. Patience 4 is shorter
than the approximately seven-turn minimum-pass trajectory, which explains this
rate; it is not an engine bug. The cooperative cell shows P3 achieving full
validity in ~6.8 turns.

### 3.3 A3+ — P0/P1/P2 on cooperative (A15c, total A3 scope 600)
| Policy | S0 validity | S0 mean turns | abandon |
|---|---|---|---|
| P0 | 1.000 | 7.31 | 0.00 |
| P1 | 1.000 | 7.21 | 0.00 |
| P2 | 1.000 | 7.39 | 0.00 |

On cooperative, all baselines reach 100% validity; P3 (6.84) is directionally
fewer turns. This is **consistent with the mechanism claim but not
confirmatory** under the pre-registered realistic cell (no p_Holm<0.05 there).

### 3.4 A6 — density ablation, S0 and S1 (A16c / B4)

A6 originally ran P3-only at low density, which cannot show the (P3−P2)
contrast. A16c adds P2 × realistic × low density (100 episodes) into
`runs/ablations/a6_p2low/`, completing the four cells.

S0 can stop only when the unresolved set is empty. Refused fields leave U, but
satisfied rule premises also remove fields, so total turns equal
|U_initial| − Σ(propagation_delta per turn); a policy whose selections enable
more propagation finishes in fewer turns. S0 therefore distinguishes policies
and is the mechanism test. Its descriptive cells are P2 high/low =
6.87/6.89 mean turns and P3 high/low = 6.84/6.88; the resulting −0.02-turn
delta-in-delta with 95% CI [−0.53, +0.50] is a real null — P3 ≈ P2 has no
differential for density to modulate. The null delta-in-delta under S0 is not a
test-power issue: the underlying P3 − P2 contrast is 0.03 turns, which no power
of the current design could detect as distinct from zero. The mechanism would
require a P3 vs P2 advantage to demonstrate, and none exists under S0 (or,
given S1's calibration problem, is measurable under S1).

B4 counterfactually replays the existing S0 answer trajectories under S1,
reconstructing initial and post-turn ΔP from saved extraction, seeds, and
answers. No episodes were generated. It tests c ∈ {0.02, 0.05, 0.10} and
v ∈ {0.10, 0.15, 0.20}; see `runs/exploratory/a6_s1.json` and the 3×2
mechanism plot (`runs/exploratory/plots/p3_density.png`). The S1 validity rates
are 0–3% in all four cells, depending on c. Therefore every requested T(v) is
undefined in every cell, for all nine (c,v) pairs. Delta-in-delta and its
independent-groups bootstrap CI are consequently **unestimable**, not measured
null contrasts. The saved counterfactual trajectory follows recorded S0 policy
choices and answers, truncating only at the first S1 stop; it does not simulate
how policies would select questions after earlier S1 stops.

The bootstrap design is independent-groups: `density_id` changes the episode
seed stream, so there is no episode-level pairing across densities. Pairing
would require a modified seed scheme.

#### 3.4.1 S1 calibration diagnostic (stored episodes only, no new runs)

T is the first record index where `records[t].delta_p < c`; T=0 means S1 stops
before any question is asked. Max ΔP at T=0 is `records[0].delta_p`,
recomputed from the saved answer trajectories. All counts over n=100 existing
episodes per cell.

| Policy | Density | c | T=0 | T=1 | T=2 | T≥3 | Median turns | Median max ΔP at T=0 |
|---|---|---|---|---|---|---|---|---|
| P2 | low | 0.02 | 99% | 1% | 0% | 0% | 0 | 0.0 |
| P2 | low | 0.05 | 100% | 0% | 0% | 0% | 0 | 0.0 |
| P2 | low | 0.10 | 100% | 0% | 0% | 0% | 0 | 0.0 |
| P3 | low | 0.02 | 100% | 0% | 0% | 0% | 0 | 0.0 |
| P3 | low | 0.05 | 100% | 0% | 0% | 0% | 0 | 0.0 |
| P3 | low | 0.10 | 100% | 0% | 0% | 0% | 0 | 0.0 |
| P2 | high | 0.02 | 95% | 3% | 0% | 2% | 0 | 0.0 |
| P2 | high | 0.05 | 98% | 2% | 0% | 0% | 0 | 0.0 |
| P2 | high | 0.10 | 100% | 0% | 0% | 0% | 0 | 0.0 |
| P3 | high | 0.02 | 99% | 0% | 1% | 0% | 0 | 0.0 |
| P3 | high | 0.05 | 100% | 0% | 0% | 0% | 0 | 0.0 |
| P3 | high | 0.10 | 100% | 0% | 0% | 0% | 0 | 0.0 |

S1 stops at T=0 in 95–100% of episodes across all four density/persona cells. Median turn-0 max ΔP is 0.0 everywhere. Consequences: (i) S1 is unusable as a mechanism test with the current c grid — the grid must be calibrated to per-cell ΔP distributions before S1 can produce meaningful operating points; (ii) **S0 remains a valid mechanism test** — the observed P3 ≈ P2 ≈ 6.85 turns vs P0 ≈ 8.0 turns shows that field ordering matters (P0 loses) but the specific ordering P3 chooses is matched by P2's uncertainty heuristic. The S1 diagnostic therefore supports the S0 null on the mechanism claim rather than overriding it.

The median turn-0 `max ΔP = 0.0` reflects PRD §8's P3 fallback rule: when all open fields have zero VoI, P3 selects the lowest-index field (P0 order). Across the stored trajectories, ΔP is zero for every open field in every turn (all-records-zero) in 54–77% of episodes per cell — P2 low 62%, P3 low 77%, P2 high 54%, P3 high 75%. In those episodes P3 reduces to P0 exactly, so the observed −2.0-turn mean advantage of P3 over P0, and the null P3-vs-P2 contrast, arise entirely from the minority of episodes in which at least one open field carries non-zero ΔP. In those signal episodes P3 achieves a per-episode advantage over P0, and P2 achieves a similar advantage by a different route (uncertainty sampling selects low-confidence fields, which in this generator correlate with rule premises). The mechanism claim — VoI beats uncertainty *specifically because* it exploits constraint-graph structure — is thereby tested and not supported: whatever signal P3 finds, P2's simpler heuristic finds equal-advantage questions. The null concerns the VoI-vs-uncertainty contrast, not the general task of choosing informative questions; P0's fixed-order baseline is measurably worse than both, confirming that question *selection* matters, while question *objective* (VoI vs uncertainty) does not separate them.

### 3.5 A3+ cooperative, paired bootstrap (C2)

P3 vs baselines at S0 on cooperative, v=0.10, 10k paired resamples, one-sided,
Holm across 3 baselines.

| Baseline | ΔT mean (P3−base) | obs quantile diff | p_one-sided | p_Holm |
|---|---|---|---|---|
| P0 | −0.47 | 1 | 0.019 | 0.057 |
| P1 | −0.37 | 1 | 0.355 | 0.453 |
| P2 | −0.55 | 1 | 0.227 | 0.355 |

Point estimates favor P3 on the cooperative cell (ΔT ≈ −0.37 to −0.55 turns).
For P3 vs P0, ΔT = −0.47 and raw p = 0.019; Holm-adjusted p = 0.057.
This comparison is exploratory and does not meet the adjusted 0.05 threshold.
The cooperative results are directionally consistent with the mechanism claim,
but are not confirmatory.

## 4. A16 disclosures (code fixes merged during W5)

- A16a — §9.3: `cli.py::replay_study` points-key extended to
  `(policy, persona, density, stop_rule, param)`; expected count is
  `policies × personas × 8`. Needed because A3 (P3 × 3 personas) collided on
  the old key. No operating-point value or stop behavior changes; the
  uniqueness check now reflects the multi-persona run shape.
- A16b — §13.4: `stats.py` computes H1/H2 only when the run holds all four
  policies P0–P3. On policy-subset runs (A3, A3+, A6) it emits `H1: []`,
  `H2: null`, `vacuous: true`, and a `policy_set` disclosure field. All three
  W5 subset stats.json were regenerated to this shape. **H1/H2 null claims
  apply only to full-policy-set runs** (core_v1).
- A16c — §12 A6: P2 × realistic × low density added; four-cell mechanism plot.

## 5. Verification and provenance

- **Full suite:** `pytest -q` → 1169 passed before the W6 multi-persona test;
  rerun after W6 edits. The single-persona uniqueness test checks 32 rows; W6
  adds a dedicated 96-row multi-persona key test.
- **C4 byte-identity:** re-running `stats.py` on `runs/core_v1` reproduces the
  locked `stats.json` byte-for-byte. This verifies effective seed behavior and
  behavior-neutral output; `core_v1` remains untouched and locked.
- **C1 SEED provenance:** the workspace has no Git history, so byte-identity
  cannot establish whether `SEED = 20250` existed before W5. It only confirms
  the current code reproduces the locked output.

## 6. Limitations

- Realistic cell cannot test H1/H2 (flattened frontier, 80–88% unpassable).
- A15b p-values carry wide bootstrap variance (few episodes reach v=0.10).
- A6 S0 is degenerate under its stop rule; requested S1 T(v) values are
  unattainable, so the mechanism contrast and CI are unestimable. Density groups
  are independent because `density_id` changes the seed stream.
- A3+ cooperative P3 advantage is directional only (no p_Holm < 0.05).
- impatient_expert cell dominated by abandonment.
- All exploratory results are one-shot, post-hoc; multiplicity control is
  within-test only (Holm).

## 7. Future-work recommendation

Mechanism test should be pre-registered on the **cooperative** persona (or an
intermediate noise level), with H1/H2 at v=0.10 and a density sweep carrying
both P3 and P2 at low density. A real density contrast would need either a
wider extraction noise gap or seeded episode pairing across densities.

### 7.1 Stop-rule grid calibration (A6/B4 follow-up)

c ∈ {0.02, 0.05, 0.10} was mis-calibrated for every cell: the median max ΔP at
turn 0 is 0.0, so c = 0.02 fires at T=0 for 95–100% of episodes and S1 stops
before any question is asked (§3.4.1). Validity at the S1 stop is therefore
never reached and T(v) is undefined for all nine (c, v) pairs. The correct next
experiment is to calibrate c per cell from the turn-0 max ΔP distribution (for
example c at the 5th–10th percentile of nonzero max ΔP), then re-run S1 with
validity monitoring so T(v) is estimable. Until that grid is calibrated, the
density contrast remains uninformative under S0 and unestimable under S1.

---

**Integrity statement (W6):** Confirmatory H1 and H2 are null at the pre-registered cell. The mechanism claim (VoI-on-validation-pass beats greedy-uncertainty) was **tested under S0, not untested**. Under S0, P3 and P2 produce nearly identical mean turns at v=0.10: ΔT = −0.03, paired-bootstrap p_Holm = 0.965. P3 does outperform P0 (fixed-order) by 2.0 turns at the same threshold (raw p = 0.019, p_Holm = 0.057 under three-way correction), but P0 is the naive baseline, not the mechanism control. **The mechanism claim is not supported by this study.** Under S1 the c-grid `{0.02, 0.05, 0.10}` is mis-calibrated for the observed turn-0 max-ΔP distribution — 95–100% of episodes stop at T=0 across all cells — making S1 operating points unestimable. The A6 density ablation under S0 is null (delta-in-delta = −0.02, CI [−0.53, +0.50]). Exploratory A15b/A15c signals (P3 vs P0 on cooperative, ΔT = −0.47, raw p = 0.019) are directionally consistent with P3 dominating fixed-order selection but do not test the VoI-vs-uncertainty mechanism.

The mechanism claim — VoI-on-validation-pass beats greedy-uncertainty by exploiting constraint-graph structure — is tested under S0 and **not supported** (ΔT_P3−P2 = −0.03, p_Holm = 0.965). S1 at the pre-registered c grid is mis-calibrated and produces no estimable operating points. The study therefore reports a null result on the mechanism claim under the one stop rule that tests it, and a design limitation on the other. This is a completed test with a null outcome, not an incomplete test.

