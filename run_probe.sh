#!/usr/bin/env bash
set -euo pipefail
cd /home/yanghc/tridi
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
exec taskset -c 16-23 /home/yanghc/miniconda3/envs/tridi-offline/bin/python -u main.py \
  -c config/env.yaml scenarios/gb_main.yaml -- \
  run.job=sample run.name=grab_test_probe sample.target=hdf5 \
  resume.checkpoint=./assets/gb_main.pth \
  dataloader.batch_size=256 dataloader.workers=0 \
  sample.mode=sample_101 'run.datasets=["grab"]' \
  sample.dataset=normal sample.repetitions=1 \
  model.cg_apply=True model.cg_scale=2.0
