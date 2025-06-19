#!/bin/bash

source ~/.bashrc
source activate eagle
which python

cd /home/shangy/TLT/eagle-train/

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

n_node=$SLURM_JOB_NUM_NODES


BASE_MODEL_PATH=${1}
MODEL_NAME=$(basename $BASE_MODEL_PATH)

BASE_DATA_PATH=${2}
DATA_PATH=${BASE_DATA_PATH}/eagle-mix
SAVE_DIR=${BASE_DATA_PATH}/eagle-processed/Eagle-Mix-${MODEL_NAME}

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    eagle_datagen.py \
    model.base_model_path=$BASE_MODEL_PATH \
    data.data_path=$DATA_PATH \
    data.save_dir=$SAVE_DIR \
    data.max_length=4096


