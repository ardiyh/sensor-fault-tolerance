#!/bin/bash
# Runs on the rtx box after the full grid (waits for ALL_DONE_2): clean-input diagnostics for PEMS-BAY.
export STC_ROOT=$HOME/paper7_SFT/stc_snapshot
PY=$HOME/paper7_SFT/venv/bin/python
cd "$HOME/paper7_SFT/SFT-code" || exit 1
until grep -q ALL_DONE_2 logs/full/FULL_STATUS; do sleep 60; done
$PY -u -m scripts.clean_bias --dataset pems-bay --device cuda --batch-size 128 > logs/full/clean_bias_pems-bay.log 2>&1
echo "clean_bias pems-bay rc=$? $(date +%T)" >> logs/full/FULL_STATUS
