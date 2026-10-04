#!/bin/bash
set -euo pipefail
cd /home/yanghc/tridi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/yanghc/miniconda3/envs/tridi-offline/bin/python
TS="taskset -c 16-23"

echo "=== [$(date '+%F %T')] preprocess TRAIN (s1-s8 @10fps) start ==="
$TS $PY -u -m tridi.preprocessing.preprocess_grab \
  'grab.subjects=["s1","s2","s3","s4","s5","s6","s7","s8"]' \
  grab.downsample="10fps" grab.split="train" \
  grab.use_decimated_obj_meshes=False grab.generate_obj_keypoints=True \
  -c ./config/env.yaml
echo "=== [$(date '+%F %T')] preprocess TRAIN done ==="

echo "=== [$(date '+%F %T')] preprocess TEST (s9-s10 @1fps) start ==="
$TS $PY -u -m tridi.preprocessing.preprocess_grab \
  'grab.subjects=["s9","s10"]' \
  grab.downsample="1fps" grab.split="test" \
  grab.use_decimated_obj_meshes=False \
  -c ./config/env.yaml
echo "=== [$(date '+%F %T')] preprocess TEST done ==="

echo "=== [$(date '+%F %T')] copy pointnext features ==="
cp -r /home/yanghc/tridi/assets/pointnext_features/ /home/yanghc/tridi/data/preprocessed/
echo "=== [$(date '+%F %T')] ALL DONE ==="
