export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TMPDIR=/local/qinghao
export TRITON_CACHE_DIR=/local/qinghao/triton_cache # Better use a non-NFS path

PROJECT_NAME=Eagle3
EXPERIMENT_NAME=Llama-3.1-8B-Instruct
EPOCHS=40
BATCH_SIZE=4
MAX_LEN=2048

BASE_MODEL_PATH=/nobackup/model/llama3.1/$EXPERIMENT_NAME
DATA_PATH=/nobackup/qinghao/dataset/eagle-processed/Eagle-Mix-$EXPERIMENT_NAME
CKPT_PATH=/nobackup/qinghao/runs/debug/$EXPERIMENT_NAME-New


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

echo "MASTER_ADDR="$MASTER_ADDR

torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    eagle3_trainer.py \
    --deepspeed_config config/deepspeed_config.json \
    --base_model_path $BASE_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $CKPT_PATH \
    --project_name $PROJECT_NAME \
    --experiment_name $EXPERIMENT_NAME \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --precision bf16 \
    --max_len $MAX_LEN  \
    --freq_map_path freq_map/llama3/freq_32768.pt

# srun -J eagle3 -N 1 --exclusive bash scripts/train_eagle3.sh