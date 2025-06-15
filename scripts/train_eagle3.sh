export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export WANDB_API_KEY='be32ec2a18acdc347b5d3029742c0ef1090a9e1e'
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TMPDIR=/local/qinghao
export TRITON_CACHE_DIR=/local/qinghao/triton_cache # Better use a non-NFS path


PROJECT_NAME=Eagle-Debug-Match
EXPERIMENT_NAME=FastRL-DS-bf16
EPOCHS=20
BATCH_SIZE=4

BASE_MODEL_PATH=/nobackup/model/llama3.1/Llama-3.1-8B-Instruct
DATA_PATH=/nobackup/qinghao/runs/eagle/eagle-data/Eagle-Mix-Llama-3.1-8B-Instruct
CKPT_PATH=/nobackup/qinghao/runs/debug  

deepspeed eagle3_trainer.py \
    --deepspeed_config config/deepspeed_config.json \
    --base_model_path $BASE_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $CKPT_PATH \
    --project_name $PROJECT_NAME \
    --experiment_name $EXPERIMENT_NAME \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --precision bf16

# srun -J eagle3 -N 1 --exclusive bash scripts/train_eagle3.sh