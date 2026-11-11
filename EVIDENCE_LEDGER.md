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
| F3 | Paper 2 metrics are unmasked: native zeros count as targets (`utils/common.py::compute_metrics`). Under that metric persistence (5.44) beats DCRNN (6.20) on clean METR-LA | `results/clean/metr-la__cpu_bs16.json` | **VERIFIED** — every result is therefore also reported with zero targets excluded (`*_nz` columns) |

## Claims

| # | Claim | Artefact | Statistic | Status |
|---|---|---|---|---|
| (filled after the pilot and after approval of the full design) | | | | |

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
