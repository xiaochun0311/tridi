#!/usr/bin/env bash
exec taskset -c 16-23 /home/yanghc/miniconda3/envs/tridi-offline/bin/python "$@"
