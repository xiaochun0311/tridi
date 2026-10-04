#!/usr/bin/env bash
set -euo pipefail
cd /home/yanghc/tridi
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
exec taskset -c 16-23 /home/yanghc/miniconda3/envs/tridi-offline/bin/python -u main.py -c config/env.yaml scenarios/gb_main.yaml -- run.name=grab_train run.job=train 'run.datasets=["grab"]' dataloader.batch_size=16 dataloader.workers=0 "$@"
