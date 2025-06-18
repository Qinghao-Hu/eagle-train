export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TMPDIR=/local/qinghao
export TRITON_CACHE_DIR=/local/qinghao/triton_cache # Better use a non-NFS path

PROJECT_NAME=Eagle3
EXPERIMENT_NAME=Llama-3.1-8B-Instruct
EPOCHS=40
BATCH_SIZE=2
MAX_LEN=4096

BASE_MODEL_PATH=/nobackup/model/llama3.1/$EXPERIMENT_NAME
DATA_PATH=/nobackup/qinghao/dataset/eagle-processed/Eagle-Mix-$EXPERIMENT_NAME
CKPT_PATH=/nobackup/qinghao/runs/debug/$EXPERIMENT_NAME

deepspeed eagle3_trainer.py \
    --deepspeed_config config/deepspeed_config.json \
    --base_model_path $BASE_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $CKPT_PATH \
    --project_name $PROJECT_NAME \
    --experiment_name $EXPERIMENT_NAME \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --precision bf16 \
    --max_len $MAX_LEN 

# srun -J eagle3 -N 2 --exclusive bash scripts/train_eagle3.sh