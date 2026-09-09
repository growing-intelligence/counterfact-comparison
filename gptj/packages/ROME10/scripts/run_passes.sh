#!/bin/bash
# Score one edited model: three sterile passes with the public harness (each alone on the GPU), then the private MRI capture.
# usage: run_passes.sh <TAG> <model_dir>      (launch detached with the backup paused; no session on the pod while it runs)
cd /workspace/cf
TAG=$1; MODEL=$2
: > logs/gpu_procs.log   # the sterile runner samples the GPU itself; no extra logger process
for split in edits controls heldout; do
  ./venv_harness/bin/python scripts/sterile_gptj.py $TAG $split $MODEL >> logs/passes_$TAG.log 2>&1
  sleep 5
done
./venv_harness/bin/python scripts/mri_capture_gptj.py $TAG $MODEL edits controls heldout > logs/mri_$TAG.log 2>&1
echo "PASSES_DONE $TAG" >> logs/baselines.log
