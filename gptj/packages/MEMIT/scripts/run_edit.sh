#!/bin/bash
# One EasyEdit run: their code, their shipped GPT-J-6B config, batch size explicit per the method's protocol.
# usage: run_edit.sh <ROME|MEMIT|AlphaEdit> [N]   (N=1 or 10 for the ROME dose points; output dir gets the N suffix)
cd /workspace/cf
METHOD=$1; N=${2:-1000}
export CF_ROOT=/workspace/cf CF_MODEL=/workspace/cf/gpt-j-6b CF_N=$N
TAG=$METHOD; [ "$N" != "1000" ] && TAG="${METHOD}${N}"
echo "[$(date -u +%H:%M:%S)] $TAG: EasyEdit run start (N=$N)" >> logs/baselines.log
./venv_easyedit/bin/python -u scripts/cf_easyedit_run.py $METHOD > logs/easyedit_$TAG.log 2>&1
grep -q CF_EASYEDIT_DONE logs/easyedit_$TAG.log && echo "EDIT_OK $TAG" >> logs/baselines.log || echo "EDIT_FAILED $TAG" >> logs/baselines.log
