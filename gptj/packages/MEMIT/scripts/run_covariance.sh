#!/bin/bash
# Covariance statistics for MEMIT and AlphaEdit (layers 3-8, EasyEdit's own cache format), one pass, then the small-sample
# verification against EasyEdit's native writer. EasyEdit environment; the fp32 model EasyEdit itself will edit.
cd /workspace/cf
export CF_ROOT=/workspace/cf CF_MODEL=/workspace/cf/gpt-j-6b CF_STATS_BT=8192
./venv_easyedit/bin/python scripts/cf_stats_onepass.py compute --layers 3,4,5,6,7,8 --sample_size 100000 --stats_dir /workspace/cf/stats > logs/cov_onepass.log 2>&1 && echo COV_COMPUTE_OK >> logs/baselines.log || { echo COV_COMPUTE_FAILED >> logs/baselines.log; exit 1; }
./venv_easyedit/bin/python scripts/cf_stats_onepass.py verify --layer 5 --sample_size 4 --out /workspace/cf/results/cov_onepass_verification.json > logs/cov_verify.log 2>&1 && echo COV_VERIFY_OK >> logs/baselines.log || echo COV_VERIFY_FAILED >> logs/baselines.log
echo COV_DONE >> logs/baselines.log
