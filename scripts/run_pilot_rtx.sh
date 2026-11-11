#!/bin/bash
# Gate: clean reproduction must pass before any fault injection.
export STC_ROOT=$HOME/paper7_SFT/stc_snapshot
PY=$HOME/paper7_SFT/venv/bin/python
cd $HOME/paper7_SFT/SFT-code
$PY -u -m scripts.reproduce_clean --dataset metr-la --device cuda --batch-size 256 > logs/repro_metr-la_cuda.log 2>&1
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open(\"results/clean/metr-la__cuda_bs256.json\"))[\"all_pass\"] else 1)"; then
  echo "GATE FAILED: reproduction did not pass, pilot not started" > logs/PILOT_STATUS; exit 1
fi
echo "gate passed, pilot running" > logs/PILOT_STATUS
$PY -u -m scripts.run_faults --dataset metr-la --faults dropout stuck --patterns random --device cuda --batch-size 256 --out results/pilot/metr-la__cuda.csv > logs/pilot_metr-la_cuda.log 2>&1 \
  && echo "PILOT DONE" > logs/PILOT_STATUS || echo "PILOT FAILED (see logs/pilot_metr-la_cuda.log)" > logs/PILOT_STATUS
