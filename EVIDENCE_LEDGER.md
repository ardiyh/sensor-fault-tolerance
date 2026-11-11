# EVIDENCE LEDGER — Paper 7 (SFT / IJRA)

Every number that reaches the manuscript gets a row here, with the artefact that backs it. A claim
with no artefact path does not go in the paper. Format follows `paper6_DAMS/DAMS-code/EVIDENCE_LEDGER.md`.

## Status legend

`PLANNED` — designed, not run · `RUN` — artefact exists · `VERIFIED` — artefact + statistics pass ·
`DEAD` — tested and did not survive

## Setup facts

| # | Fact | Artefact | Status |
|---|---|---|---|
| F1 | Clean-input MAE of all five learned METR-LA models matches Paper 2 `results_metr-la.json` within 0.005% and Paper 6 dumps exactly; Paper 2's "SeasonalNaive" (7.4325) is a last-window (lag-12) naive and reproduces exactly | `results/clean/metr-la__cpu_bs16.json`, `__cpu_bs256.json`, `__mps_bs256.json` | **VERIFIED** (CPU and MPS agree to 3e-5 relative) |
| F2 | Native missing data (zeros): METR-LA 8.11% of all readings, 12.10% of the test split, 666 test timesteps with every sensor at zero; PEMS-BAY 0.003% / 0.006% / 0 | `scripts/native_zeros.py` → `results/native_zeros.json` | **VERIFIED**. Also: 90 of 207 METR-LA sensors exceed 10% zeros in the test split (max 44.5%); 12.14% of test target entries are zero |
| F3 | Paper 2 metrics are unmasked: native zeros count as targets (`utils/common.py::compute_metrics`). Under that metric persistence (5.44) beats DCRNN (6.20) on clean METR-LA | `results/clean/metr-la__cpu_bs16.json` | **VERIFIED** — every result is therefore also reported with zero targets excluded (`*_nz` columns). Excluding zeros: DCRNN 5.07 < persistence 5.15 < students 7.78–8.06 |
| F4 | Paper 2's "DCRNN" has no spatial mixing: `models/dcrnn.py:50` `einsum('nn,bnc->bnc')` takes the diagonal of the transition matrix, which is zero (self-loops removed), so all diffusion terms vanish. Zeroing all other 206 sensors changes a sensor's forecast by exactly 0.0 | `scripts/check_dcrnn_mixing.py` → `results/dcrnn_mixing_metr-la.json` | **VERIFIED**. Same code in `paper3_QGD/QGD-code/src/models/dcrnn.py`. Paper 7 must not call this checkpoint DCRNN; decision pending (relabel as GRU encoder–decoder and/or train a corrected DCRNN) |
| F5 | Clean and faulted results are device-independent: CPU (Mac) vs CUDA (rtx) pilot rows agree to 8e-7 relative; masks identical | `results/pilot/metr-la__light.csv` vs `results/pilot/metr-la__cuda.csv` | **VERIFIED** |

## Claims

Pilot = METR-LA, random pattern, dropout and stuck, fractions 5–50%, 5 mask seeds, one checkpoint
per model. Source for all P rows: `results/pilot/metr-la__cuda.csv`, `metr-la__summary.{csv,json}`.

| # | Claim | Artefact | Statistic | Status |
|---|---|---|---|---|
| P1 | Under sensor dropout, both students raise error on HEALTHY sensors (contamination): healthy-target MAE (zeros excluded) ×1.12–1.18 at 5% faulty, ×1.73–1.88 at 20%, ×2.77–3.50 at 50%; non-spatial models (persistence, seasonal naive, Paper 2 "DCRNN") stay at exactly ×1.000 | summary csv `ratio_healthy_mae_nz_mean` | Wilcoxon healthy-node MAE fault vs clean: p<0.05 in 5/5 seeds at every fraction for all four students; 0/5 for the non-spatial models | RUN (pilot; single checkpoint per model) |
| P2 | Students partly compensate on FAULTY sensors: faulty-target MAE 14.4–16.5 mph at 5% dropout vs 51.0–57.6 for the non-spatial models, which forecast near zero for a zeroed sensor | summary csv `faulty_mae_nz_mean` | mean over 5 mask seeds | RUN (pilot) |
| P3 | Net effect at low fractions: at 5% dropout the students' all-target MAE (10.2–10.8) stays above persistence (7.59) and "DCRNN" (8.09); at 50% the Student-Attn task-only (25.4±1.8) is lowest | summary csv `all_mae_mean/std` | mean ± std over 5 mask seeds | RUN (pilot) |
| P4 | KD changes robustness non-monotonically: KD students degrade LESS at 5–10% dropout (Attn healthy +11.5% vs +14.5%, node win rate 0.84) and MORE at ≥20–30% (Attn healthy +199% vs +145% at 50%; MLP +207% vs +173%); same direction for both pairs | summary json `kd_vs_task_only` | node-paired Wilcoxon, pooled over 5 mask seeds; does NOT account for training-seed variance (one checkpoint per arm) | RUN — must be re-tested with 5 vs 5 Student-Attn checkpoints from Drive before any claim |
| P5 | Stuck as implemented (frozen at the first input sample, ≤55 min stale) is mild: students ≤ +9% all-target MAE at 50% faulty; persistence ×1.32 | summary csv | mean over 5 mask seeds | RUN — near-flat; a longer freeze is needed if stuck stays in the paper |

## Non-overlap audit (must pass before submission)

| Prior paper | Owned claim | This paper must not restate |
|---|---|---|
| Paper 2 `STC-paper/icicelb.tex` | AT-KD method, compression ratio, accuracy–latency Pareto | No latency table, no compression-ratio claim, no re-derivation of the Pareto front. Students are used as-is and cited third-person |
| Paper 3 QGD | Teacher-quality gate; when distillation hurts | No gating, no claim about when KD helps clean accuracy |
| Paper 4 | Chronos-2 zero-shot benchmark, context length | No TSFM benchmarking; the teacher is not evaluated |
| Paper 6 DAMS | Cost-loss decision value, model-selection rule, INT8 quantisation, CPU latency | No decision-value analysis, no quantisation, no latency numbers |

## Pre-submission checks

- [ ] No "our prior work" identity leaks; Paper 2 cited third-person
- [ ] "KD+AT" described accurately: the attention-transfer term is numerically negligible
      (≤ 2/N² per the 9 Oct 2026 audit), so any KD effect is an output-distillation effect
- [ ] Word count ≤ 4,000; 8 pages; ≥ 25 IEEE references, ≥ 15 recent journal articles; self-citations ≤ 5
- [ ] Abstract 100–200 words, past tense, no citations
- [ ] Figure 1 is the sensor → roadside unit → forecaster → TMC chain with the fault entry points
- [ ] Every table number traceable to a row above
