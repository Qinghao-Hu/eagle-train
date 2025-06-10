export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export WANDB_API_KEY='be32ec2a18acdc347b5d3029742c0ef1090a9e1e'
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TMPDIR=/local/qinghao

EPOCHS=10
BATCH_SIZE=4
PROJECT_NAME=Eagle-Debug-Match
EXPERIMENT_NAME=FastRL-DS-bf16

BASE_MODEL_PATH=/nobackup/model/llama3.1/Llama-3.1-8B-Instruct
DATA_PATH=/nobackup/qinghao/runs/eagle/eagle-sft-data/sharegpt_0_67999_mufp16_llama31_8B
CKPT_PATH=/nobackup/qinghao/runs/debug

deepspeed eagle_trainer.py \
    --deepspeed_config deepspeed_config.json \
    --base_model_path $BASE_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $CKPT_PATH \
    --project_name $PROJECT_NAME \
    --experiment_name $EXPERIMENT_NAME \
    --wandb_project "$PROJECT_NAME" \
    --wandb_name "$EXPERIMENT_NAME" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --precision bf16

# srun -J eagle -N 1 --exclusive bash eagle-train/train_eagle2.sh