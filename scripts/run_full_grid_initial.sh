#!/bin/bash
# Copied from the rtx run directory (logs/) on 12 Nov 2026; this is the exact script that was run.
# Its METR-LA light run and all three PEMS-BAY runs failed with CUDA OOM (six processes on 8 GB);
# scripts/run_full_grid_rtx.sh reran them at smaller batch sizes. METR-LA gru/dcrnn finished here.
export STC_ROOT=$HOME/paper7_SFT/stc_snapshot
PY=$HOME/paper7_SFT/venv/bin/python
cd $HOME/paper7_SFT/SFT-code
LIGHT="persistence seasonal_naive mlp_task_p2 mlp_kd_p2 attn_task_rev1 attn_kd_rev1 mlp_task_p7 mlp_aug_p7 attn_task_p7 attn_aug_p7"
run() { ds=$1; part=$2; shift 2
  $PY -u -m scripts.run_grid --dataset $ds --groups "$@" --device cuda --batch-size 1024 \
      --out results/full/${ds}__${part}.csv > logs/full/${ds}__${part}.log 2>&1
  echo "$ds $part rc=$? $(date +%T)" >> logs/full/FULL_STATUS; }
for ds in metr-la pems-bay; do
  run $ds light $LIGHT &
  run $ds gru gru_p2 &
  run $ds dcrnn dcrnn_fixed &
done
wait
echo ALL_DONE >> logs/full/FULL_STATUS
