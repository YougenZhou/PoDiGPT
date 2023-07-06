#!/bin/bash

if [ $# == 1 ]; then
    config_file=$1
    source ${config_file}
elif [ $# > 1 ]; then
  echo "usage: bash $0 [config_file]"
  exit -1
fi

export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

python  ./pdgpt/scripts/finetune.py \
  --config ${config}