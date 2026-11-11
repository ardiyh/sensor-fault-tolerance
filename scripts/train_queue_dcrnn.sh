#!/bin/bash
# Copied from the rtx run directory (logs/) on 12 Nov 2026; this is the exact script that was run.
export STC_ROOT=$HOME/paper7_SFT/stc_snapshot
PY=$HOME/paper7_SFT/venv/bin/python
cd $HOME/paper7_SFT/SFT-code
for ds in metr-la pems-bay; do
  echo "start $ds $(date)" >> logs/train/QUEUE_DCRNN
  $PY -u scripts/train_dcrnn.py --dataset $ds --normalize --batch-size 64 --epochs 50 --checkpoint-tag p7 \
      --results-json results/train/dcrnn_fixed_${ds}.json > logs/train/dcrnn_fixed_${ds}.log 2>&1
  echo "done $ds rc=$? $(date)" >> logs/train/QUEUE_DCRNN
done
echo ALL_DONE >> logs/train/QUEUE_DCRNN
