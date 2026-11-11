#!/bin/bash
# Full grid on the rtx box (RTX 3070, 8 GB). Usage: scripts/run_full_grid_rtx.sh <pid> [<pid> ...]
# The PIDs are the METR-LA GRU/DCRNN grid runs already in progress; the PEMS-BAY recurrent runs
# wait for them so that GPU memory stays below 8 GB. PEMS-BAY recurrent models use 3 mask seeds
# and the front-end on dropout only (compute budget); every other run is the full grid.
export STC_ROOT=$HOME/paper7_SFT/stc_snapshot
PY=$HOME/paper7_SFT/venv/bin/python
cd "$HOME/paper7_SFT/SFT-code" || exit 1
LIGHT="persistence seasonal_naive mlp_task_p2 mlp_kd_p2 attn_task_rev1 attn_kd_rev1 mlp_task_p7 mlp_aug_p7 attn_task_p7 attn_aug_p7"
WAIT_PIDS="$*"

run() {
  ds=$1; part=$2; bs=$3; shift 3
  $PY -u -m scripts.run_grid --dataset "$ds" --device cuda --batch-size "$bs" \
      --out "results/full/${ds}__${part}.csv" "$@" > "logs/full/${ds}__${part}.log" 2>&1
  echo "$ds $part rc=$? $(date +%T)" >> logs/full/FULL_STATUS
}

( run metr-la light 256 --groups $LIGHT
  run pems-bay light 256 --groups $LIGHT ) &
( while kill -0 $WAIT_PIDS 2>/dev/null; do sleep 30; done
  run pems-bay gru 512 --groups gru_p2 --seeds 3 --repair-faults dropout &
  run pems-bay dcrnn 512 --groups dcrnn_fixed --seeds 3 --repair-faults dropout &
  wait ) &
wait
echo ALL_DONE_2 >> logs/full/FULL_STATUS
