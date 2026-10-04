#!/usr/bin/env bash
set -euo pipefail
cd /home/yanghc/tridi
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=/home/yanghc/miniconda3/envs/tridi-offline/bin/python
for split in train test; do
  fps=10fps
  keypoints=True
  if [ "$split" = test ]; then fps=1fps; keypoints=False; fi
  taskset -c 16-23 "$PYTHON" -u -m tridi.preprocessing.preprocess_behave_30fps -c config/env.yaml -- behave.full_anno_path=data/raw/behave/behave-30fps-params-v1 behave.split="$split" behave.downsample="$fps" behave.generate_obj_keypoints="$keypoints"
done
echo BEHAVE_PREPROCESS_COMPLETE
