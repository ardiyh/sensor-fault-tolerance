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
| F1b | PEMS-BAY clean reproduction: DCRNN-labelled GRU and both Student-Attn checkpoints match Paper 2 within 1.04e-4 relative (max: Student-Attn baseline); Paper 2's PEMS-BAY Student-MLP checkpoints are absent from the snapshot, the backup copies used instead match Paper 2 exactly (F7). Manuscript states "within 0.011%" for both datasets | `results/clean/pems-bay__cuda_bs256.json` | **VERIFIED** |
| F6 | Corrected DCRNN (`src/dcrnn_fixed.py`, einsum `nm,bmc->bnc`, otherwise Paper 2 code) trained here: unmasked MAE 5.633 METR-LA / 1.807 PEMS-BAY vs 6.197 / 2.107 for the Paper 2 checkpoint; spatial mixing verified (zeroing neighbours changes forecasts by 9–49 mph) | `results/train/dcrnn_fixed_{metr-la,pems-bay}.json`, `results/dcrnn_fixed_mixing.json`, rtx `logs/train/dcrnn_fixed_*.log` (early stop at epoch 43 / 34; batch 64, ≤50 epochs, patience 15) | **VERIFIED** |
| F7 | Seed checkpoints for RQ4: rev1 matched sets (Student-Attn task `rev1parity` s0–4, KD+AT `rev1batchF_matchedkdat` s0–4) are complete for both datasets but do not reproduce Paper 2's multiseed table values; they are used as matched 5-vs-5 arms. PEMS-BAY Student-MLP task/KD (backup) match Paper 2 exactly (3.8087 / 3.7936) | `results/seed_checkpoints.json` | **VERIFIED** (provenance documented; not a claim of the paper) |
| F8 | Clean-input diagnostics (METR-LA): every learned model under-predicts non-zero speeds (mean signed error −1.1 to −2.2 mph for students; 62–68% of non-zero targets under-predicted); 2.6% of non-zero targets come from windows whose own input holds ≥6 native zeros, with MAE 22.9–26.3 mph for every learned model vs 7.0–7.5 (students) on windows without zeros | `scripts/clean_bias.py` → `results/clean_bias_metr-la.json` | **VERIFIED** — explains M5 (bias lowers student error) and M8/M9 (remedies lower the clean error) |
| F5 | Clean and faulted results are device-independent: CPU (Mac) vs CUDA (rtx) pilot rows agree to 8e-7 relative; masks identical | `results/pilot/metr-la__light.csv` vs `results/pilot/metr-la__cuda.csv` | **VERIFIED** |

## Claims

Pilot = METR-LA, random pattern, dropout and stuck, fractions 5–50%, 5 mask seeds, one checkpoint
per model. Source for all P rows: `results/pilot/metr-la__cuda.csv`, `metr-la__summary.{csv,json}`.

| # | Claim | Artefact | Statistic | Status |
|---|---|---|---|---|
| P1 | Under sensor dropout, both students raise error on HEALTHY sensors (contamination): healthy-target MAE (zeros excluded) ×1.12–1.18 at 5% faulty, ×1.73–1.88 at 20%, ×2.77–3.50 at 50%; non-spatial models (persistence, seasonal naive, Paper 2 "DCRNN") stay at exactly ×1.000 | summary csv `ratio_healthy_mae_nz_mean` | Wilcoxon healthy-node MAE fault vs clean: p<0.05 in 5/5 seeds at every fraction for all four students; 0/5 for the non-spatial models | RUN (pilot; single checkpoint per model) |
| P2 | Students partly compensate on FAULTY sensors: faulty-target MAE 14.4–16.5 mph at 5% dropout vs 51.0–57.6 for the non-spatial models, which forecast near zero for a zeroed sensor | summary csv `faulty_mae_nz_mean` | mean over 5 mask seeds | RUN (pilot) |
| P3 | Net effect at low fractions: at 5% dropout the students' all-target MAE (10.2–10.8) stays above persistence (7.59) and "DCRNN" (8.09); at 50% the Student-Attn task-only (25.4±1.8) is lowest | summary csv `all_mae_mean/std` | mean ± std over 5 mask seeds | RUN (pilot) |
| P4 | ~~KD changes robustness non-monotonically~~: KD students degrade LESS at 5–10% dropout (Attn healthy +11.5% vs +14.5%, node win rate 0.84) and MORE at ≥20–30% (Attn healthy +199% vs +145% at 50%; MLP +207% vs +173%); same direction for both pairs | summary json `kd_vs_task_only` | node-paired Wilcoxon, pooled over 5 mask seeds; does NOT account for training-seed variance (one checkpoint per arm) | **DEAD** — not reproduced with 5 vs 5 matched checkpoints (M6): the healthy-sensor difference is inside the training-seed spread at every fraction. Single-checkpoint KD comparisons are not reliable |
| P5 | Stuck as implemented (frozen at the first input sample, ≤55 min stale) is mild: students ≤ +9% all-target MAE at 50% faulty; persistence ×1.32 | summary csv | mean over 5 mask seeds | RUN — near-flat; superseded by the added 3-h `stuck_long` fault (M3) |

## Full grid — METR-LA

Source: `results/full/metr-la__{light,gru,dcrnn}.csv` + `.pernode.npz` (rtx, 12 Nov 2026), aggregated
by `scripts/analyze_full.py` → `results/full/metr-la__groups.csv`, `metr-la__tests.json`. Metric:
MAE on non-zero targets (mph). 5 mask seeds × (5 training seeds per student set | 1 checkpoint).

| # | Claim | Artefact | Statistic | Status |
|---|---|---|---|---|
| M1 | Clean MAE: DCRNN 4.58, recurrent no-mixing 5.07, persistence 5.15, students 7.80–8.09, seasonal naive 10.34. The compact students are less accurate than persistence on clean input | `metr-la__groups.csv` (fault=none, repair=0); `results/clean_bias_metr-la.json` | mean (± std over 5 training seeds) | **VERIFIED** — must be stated in the paper |
| M2 | Dropout at 20% random: student MAE ×1.81–1.99 (Attn task 8.04 → 15.31), persistence ×2.99 (5.15 → 15.37) | `metr-la__groups.csv` `ratio_all_mae_nz_mean` | mean ± std over 25 runs | **VERIFIED** |
| M3 | Stuck (≤55 min) ×1.03, 3-h stuck ×1.08–1.10, noise ×1.012–1.016 for students at 20% random; persistence ×1.11 / ×1.29 / ×1.20 | `metr-la__groups.csv` | mean over runs | **VERIFIED** |
| M4 | Contamination: random dropout raises healthy-sensor MAE of every student in every run (R_H 1.12–1.16 at 5%, 1.68–1.83 at 20%, 3.0–3.5 at 50%); non-spatial models R_H = 1 | `metr-la__tests.json` `contamination` | Wilcoxon over healthy sensors, p<0.05 in 100% of runs (random), 56–100% (clustered) | **VERIFIED** |
| M4b | Clustered outages spread less but are compensated worse: at 20% dropout R_H 1.11–1.25 (vs 1.68–1.83 random) while faulty-sensor MAE 38.6–43.6 mph (vs 19.1–21.4); non-spatial models 56–57 mph | `metr-la__groups.csv` | mean over runs | **VERIFIED** |
| M5 | Positive bias (+0.5σ) LOWERS student MAE (×0.958–0.992 at 5–30%) while persistence rises ×1.06–1.37; cause: students under-predict (F8) | `metr-la__groups.csv`, `results/clean_bias_metr-la.json` | mean over runs | **VERIFIED** — a calibration artefact, not robustness |
| M6 | Distillation (Student-Attn KD+AT vs task-only, 5 vs 5): clean MAE 7.87±0.06 vs 8.04±0.10 (Mann–Whitney p=0.016). Healthy-sensor relative degradation under dropout identical (20%: +90.1% vs +90.3%, p=1.0; all fractions p≥0.55). Faulty-sensor relative degradation slightly larger with KD (+6–15 points; p=0.008 at 10%, 0.016 at 30%, ≥0.15 elsewhere) because the KD clean baseline is lower; absolute faulty MAE equal (20%: 20.47 vs 20.84, p=0.69) | `metr-la__tests.json` `pairs.kd_attn`; absolute per-seed MAE from `metr-la__light.csv` | seed-level Mann–Whitney (5 vs 5); node-level Wilcoxon and bootstrap CI reported but not used as the unit of inference | **VERIFIED** — null on contamination |
| M7 | Student-MLP KD (one checkpoint) degrades more than all 5 retrained task-only MLPs under dropout ≥10% (×1.94 vs 1.78–1.83 at 20%), but the Paper 2 task-only MLP is also outside that band in 36% of settings | `metr-la__tests.json` `mlp_kd_vs_task_band`, `mlp_task_p2_vs_task_band` | band = min–max of 5 seeds | RUN — not attributable to KD (n=1) |
| M8 | Repair front-end: Student-Attn task at 20% random dropout 15.31 → 7.72 (clean 8.04); on clean input 8.04 → 7.57 because it fills native outages (12.1% of healthy sensor-windows flagged); clustered 50% dropout 26.21 → 8.35 | `metr-la__groups.csv` repair=1, `metr-la__tests.json` `repair` | mean over runs | **VERIFIED** |
| M9 | Dropout-trained Student-Attn vs its retrained task-only twin (5 vs 5): R_H at 20% dropout 1.01 vs 1.83 (all-sensor ratio 1.04 vs 1.99), faulty MAE 9.40 vs 21.43, all dropout/stuck settings p=0.008; clean non-zero MAE 7.69 vs 7.94 (p=0.008) but unmasked MAE unchanged (9.26 vs 9.24, p=1.0); cost: noise sensitivity of healthy sensors +1.9% vs +1.0% at 20% (p=0.056), +4.3% vs +1.9% at 50% (p=0.016). Under CLUSTERED dropout ≥20% it does not reduce contamination (healthy p=0.31–0.84; at 20% +18.0% vs +14.1%) | `metr-la__tests.json` `pairs.aug_attn`, `metr-la__light.csv` | seed-level Mann–Whitney | **VERIFIED** |
| M10 | Recurrent references at 20% random (METR-LA): DCRNN ×3.03 dropout (4.58 → 13.88), ×1.12 stuck, ×1.32 long stuck, ×1.05 noise, ×1.14 bias; recurrent no-mixing ×2.78 / 1.11 / 1.27 / 1.07 / 1.15; with persistence these non-student models lose 5–32% under stuck/noise/bias vs ≤10% for students. Under dropout all non-seasonal forecasters end at 13.9–15.8 mph | `metr-la__groups.csv` (groups dcrnn_fixed, gru_p2) | mean over 5 masks | **VERIFIED** |
| M11 | DCRNN contaminates mildly and compensates little: R_H 1.11 at 20% random dropout (1.03–1.32 over 5–50%; 1.04–1.51 clustered), Wilcoxon p<0.05 in every run; faulty-sensor MAE 49.6 mph (non-spatial 50.7–56.9, students 19.1–21.4). Front-end: DCRNN 13.88 → 5.21 at 20% dropout | `metr-la__tests.json` `contamination`, `metr-la__groups.csv` | Wilcoxon per run | **VERIFIED** |

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
