#!/usr/bin/env bash
set -euo pipefail
cd /home/yanghc/tridi/tridi/utils/smplx2smplh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY=/home/yanghc/miniconda3/envs/tridi-offline/bin/python
G=/home/yanghc/tridi/data/raw/grab
S=/home/yanghc/tridi/data/smplx_models
LOG=/home/yanghc/tridi

# GPU 0: s1 s8 s7 s5  (~49k frames)
nohup env CUDA_VISIBLE_DEVICES=0 taskset -c 16-19 "$PY" -u convert_grab_all.py \
  -g "$G" -s "$S" --subjects s1 s8 s7 s5 \
  > "$LOG/convert_gpu0.log" 2>&1 &
echo $! > "$LOG/convert_gpu0.pid"

# GPU 1: s6 s10 s9 s3 s4 s2  (~48k frames)
nohup env CUDA_VISIBLE_DEVICES=1 taskset -c 20-23 "$PY" -u convert_grab_all.py \
  -g "$G" -s "$S" --subjects s6 s10 s9 s3 s4 s2 \
  > "$LOG/convert_gpu1.log" 2>&1 &
echo $! > "$LOG/convert_gpu1.pid"

echo "launched PIDs: $(cat "$LOG/convert_gpu0.pid") $(cat "$LOG/convert_gpu1.pid")"
