# Sensor fault tolerance of compact graph forecasters

Experiment and analysis code for:

> **Sensor fault tolerance of compact graph forecasters for intelligent transportation**
> Ardi Yusri Hilmi, Edi Winarko, Moh Edi Wibowo, Andi Dharmawan
> Department of Computer Science and Electronics, Universitas Gadjah Mada

The study injects dropout, stuck, noise, and bias faults into 5–50% of the sensors of METR-LA and
PEMS-BAY, placed at random or in contiguous clusters. It then measures two errors separately: the
error on the faulty sensors, and the error spread to the healthy ones. It evaluates distilled and
task-only ST-Chronos students, a diffusion convolutional recurrent network (DCRNN), a recurrent
model without spatial mixing, persistence, and a seasonal naive forecast. It also evaluates two
remedies: a training-free neighbour-imputation front-end, and retraining with random sensor
dropout.

All errors are reported on non-zero targets (`*_nz` columns), because METR-LA records outages as
zero speed; 12.1% of its test readings are zero. The unmasked errors are kept alongside.

## Layout

| Path | Contents |
|---|---|
| `src/faults.py` | Fault models (dropout, stuck, long stuck, noise, bias) and random / clustered placement |
| `src/repair.py` | Front-end: flags constant or zero-filled windows, imputes from unflagged neighbours |
| `src/augment.py` | Sensor-dropout augmentation used for retraining |
| `src/dcrnn_fixed.py` | DCRNN with the diffusion step applied over neighbours (see below) |
| `src/runner.py`, `src/metrics.py`, `src/stats.py` | Evaluation loop, healthy/faulty-sensor metrics, tests |
| `configs/models_<dataset>.json` | Checkpoints and settings of every evaluated model |
| `scripts/` | Entry points; the `*.sh` files are the exact queues used for the reported runs |
| `results/` | Raw results (CSV/JSON) behind every number in the paper |
| `EVIDENCE_LEDGER.md` | One row per reported number, with the file and statistic that back it, including claims that did not survive |
| `tests/` | Unit tests (`pytest`) |

## Requirements

The reported runs used Python 3.10, PyTorch 2.3.1 (CUDA 12.1), PyTorch Geometric 2.8.0, NumPy 2.2,
SciPy 1.15, pandas 2.3, and matplotlib 3.10 on an NVIDIA RTX 3070 (8 GB).

The student checkpoints and their model and data-loading code come from the ST-Chronos codebase,
which is not part of this repository. `src/stc_bridge.py` is the only module that imports it; set
`STC_ROOT` to a copy that contains `models/`, `utils/`, `data/` (METR-LA and PEMS-BAY in the DCRNN
release format) and the checkpoints listed in `configs/`. Nothing in this repository modifies that
codebase.

## Reproducing the results

```bash
export STC_ROOT=/path/to/STC-code
pytest                                                    # unit tests
python -m scripts.reproduce_clean --dataset metr-la       # clean errors vs the published ones
python -m scripts.run_grid --dataset metr-la --device cuda --out results/full/metr-la__light.csv \
       --groups persistence seasonal_naive mlp_task_p2 mlp_kd_p2 attn_task_rev1 attn_kd_rev1 \
                mlp_task_p7 mlp_aug_p7 attn_task_p7 attn_aug_p7
python -m scripts.analyze_full --dataset metr-la          # groups.csv and tests.json
python -m scripts.make_tables                             # Table 2
python -m scripts.make_figures                            # Figure 2
```

`scripts/run_full_grid_rtx.sh`, `scripts/train_queue_dcrnn.sh`, and
`scripts/train_queue_students.sh` list the exact commands, batch sizes, and seeds of the reported
runs. `make_tables` and `make_figures` write into a sibling `SFT-paper/` directory (the
manuscript). The per-node error arrays (`*.pernode.npz`, about 70 MB) that `analyze_full` reads
are not tracked; `run_grid` writes them next to each CSV.

| Paper element | Produced by | From |
|---|---|---|
| Table 2 | `scripts/make_tables.py` | `results/full/<dataset>__groups.csv` |
| Figure 2 | `scripts/make_figures.py` | `results/full/<dataset>__groups.csv` |
| Test statistics in Section 4 | `scripts/analyze_full.py` | `results/full/<dataset>__tests.json` |
| Reproduction tolerance | `scripts/reproduce_clean.py` | `results/clean/*.json` |
| Native outages | `scripts/native_zeros.py` | `results/native_zeros.json` |
| Signed error and outage windows on clean input | `scripts/clean_bias.py` | `results/clean_bias_<dataset>.json` |
| Corrected DCRNN training | `scripts/train_dcrnn.py` | `results/train/dcrnn_fixed_<dataset>.json` |

## Note on the DCRNN baseline

The DCRNN checkpoint distributed with ST-Chronos computes its diffusion step as
`einsum('nn,bnc->bnc', P, x)`. This reads only the diagonal of the transition matrix, and that
diagonal is zero, so the model has no spatial mixing; `scripts/check_dcrnn_mixing.py` shows this.
That checkpoint is evaluated here as a recurrent model without mixing. `src/dcrnn_fixed.py` uses
`einsum('nm,bmc->bnc', P, x)` and was trained from scratch for this study; it is the model labelled
DCRNN in the paper.

## License

MIT (see `LICENSE`).
