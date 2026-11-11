#!/bin/bash
# Copied from the rtx run directory (logs/) on 12 Nov 2026; this is the exact script that was run.
export STC_ROOT=$HOME/paper7_SFT/stc_snapshot
PY=$HOME/paper7_SFT/venv/bin/python
cd $HOME/paper7_SFT/SFT-code
for ds in metr-la pems-bay; do for s in 0 1 2 3 4; do
  for aug in 0 1; do
    if [ $aug = 1 ]; then tag=p7augs$s; extra="--sensor-dropout 0.5"; else tag=p7s$s; extra=""; fi
    $PY -u scripts/train_student_mlp.py --dataset $ds --normalize --baseline --batch-size 256 --epochs 50 \
        --seed $s --checkpoint-tag $tag $extra > logs/train/mlp_${ds}_${tag}.log 2>&1
    echo "mlp $ds $tag rc=$? $(date +%T)" >> logs/train/QUEUE_STUDENTS
    $PY -u scripts/train_student_attn.py --dataset $ds --normalize --baseline --batch-size 256 --epochs 50 \
        --seed $s --no-compile --checkpoint-tag $tag $extra > logs/train/attn_${ds}_${tag}.log 2>&1
    echo "attn $ds $tag rc=$? $(date +%T)" >> logs/train/QUEUE_STUDENTS
  done
done; done
echo ALL_DONE >> logs/train/QUEUE_STUDENTS
