# bash scripts/train_eagle3.sh /home/jerguo/dataset/qwen2.5/Qwen2.5-1.5B /home/jerguo/dataset/datasets/tlt

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TMPDIR=/local/jerguo/tlt/tmp
export TRITON_CACHE_DIR=/local/jerguo/tlt/tmp/triton_cache # Better use a non-NFS path

mkdir -p /local/jerguo/tlt/tmp/triton_cache

PROJECT_NAME=Eagle3
EPOCHS=40
BATCH_SIZE=2
MAX_LEN=4096

BASE_MODEL_PATH=${1}
MODEL_NAME=$(basename $BASE_MODEL_PATH)
EXPERIMENT_NAME=${MODEL_NAME}

BASE_DATA_PATH=${2}
DATA_PATH=${BASE_DATA_PATH}/eagle-processed/Eagle-Mix-${MODEL_NAME}
CKPT_PATH=/home/jerguo/runs/tlt/eagle-checkpoints/$EXPERIMENT_NAME


FREQ_MAP_PATH="freq_map/$MODEL_NAME/freq_32768.pt"

# Detect if the FREQ_MAP_PATH is valid
if [ ! -f "$FREQ_MAP_PATH" ]; then
    echo "FREQ_MAP_PATH: $FREQ_MAP_PATH does not exist, using default freq_map_path"
    # Determine freq_map_path from BASE_MODEL_PATH
    if [[ "$BASE_MODEL_PATH" == *"Llama-3"* ]]; then
        FREQ_MAP_PATH="freq_map/llama3/freq_32768.pt"
    elif [[ "$BASE_MODEL_PATH" == *"Qwen2.5"* ]]; then
        FREQ_MAP_PATH="freq_map/qwen2.5/freq_32768.pt"
    elif [[ "$BASE_MODEL_PATH" == *"Qwen3"* ]]; then
        FREQ_MAP_PATH="freq_map/qwen3/freq_32768.pt"
    else
        echo "Could not determine model type from BASE_MODEL_PATH for freq_map_path"
        exit 1
    fi
fi



echo "Using FREQ_MAP_PATH: $FREQ_MAP_PATH"


torchrun --nnodes=1 --nproc_per_node=8 \
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
    --max_len $MAX_LEN \
    --save_steps 8000 \
    --freq_map_path $FREQ_MAP_PATH

# srun -J eagle3 -N 4 --exclusive bash scripts/train_eagle3.sh