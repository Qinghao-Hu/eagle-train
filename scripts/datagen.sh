# srun -J datagen -N 3 --exclusive bash scripts/datagen.sh

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID


MODEL_NAME=Llama-3.1-8B-Instruct
BASE_MODEL_PATH=/nobackup/model/llama3.1/${MODEL_NAME}
DATA_PATH=/nobackup/qinghao/dataset/eagle-mix
SAVE_DIR=/nobackup/qinghao/dataset/eagle-processed/Eagle-Mix-${MODEL_NAME}

torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    eagle_datagen.py \
    model.base_model_path=$BASE_MODEL_PATH \
    data.data_path=$DATA_PATH \
    data.save_dir=$SAVE_DIR \
    data.max_length=2048



# torchrun --standalone --nnodes=1 --nproc_per_node=8 -m fastrl.trainer.eagle_datagen_sft \
#     data.data_path=/nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M \
#     data.save_dir=$SAVE_DIR/OpenThoughts2-1M-Qwen-7B-Instruct \
#     model.base_model_path=/nobackup/model/qwen2.5/Qwen2.5-7B-Instruct

# torchrun --standalone --nnodes=1 --nproc_per_node=8 -m fastrl.trainer.eagle_datagen_sft \
#     data.data_path=/nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M \
#     data.save_dir="${SAVE_DIR}/OpenThoughts2-1M-Qwen-32B" \
#     model.base_model_path=/nobackup/model/qwen2.5/Qwen2.5-32B


# CUDA_VISIBLE_DEVICES=0,1 python fastrl/trainer/eagle_datagen_sft_70B.py \
#     data.data_path=/nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M \
#     data.save_dir="${SAVE_DIR}/OpenThoughts2-1M-Llama-3.3-70B-Instruct" \
#     model.base_model_path=/nobackup/model/llama3.3/Llama-3.3-70B-Instruct \
#     data.process_rank=0 &


# CUDA_VISIBLE_DEVICES=2,3 python fastrl/trainer/eagle_datagen_sft_70B.py \
#     data.data_path=/nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M \
#     data.save_dir="${SAVE_DIR}/OpenThoughts2-1M-Llama-3.3-70B-Instruct" \
#     model.base_model_path=/nobackup/model/llama3.3/Llama-3.3-70B-Instruct \
#     data.process_rank=1 &

# CUDA_VISIBLE_DEVICES=4,5 python fastrl/trainer/eagle_datagen_sft_70B.py \
#     data.data_path=/nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M \
#     data.save_dir="${SAVE_DIR}/OpenThoughts2-1M-Llama-3.3-70B-Instruct" \
#     model.base_model_path=/nobackup/model/llama3.3/Llama-3.3-70B-Instruct \
#     data.process_rank=2 &

# CUDA_VISIBLE_DEVICES=6,7 python fastrl/trainer/eagle_datagen_sft_70B.py \
#     data.data_path=/nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M \
#     data.save_dir="${SAVE_DIR}/OpenThoughts2-1M-Llama-3.3-70B-Instruct" \
#     model.base_model_path=/nobackup/model/llama3.3/Llama-3.3-70B-Instruct \
#     data.process_rank=3


# CUDA_VISIBLE_DEVICES=0,1,2,3 python fastrl/trainer/eagle_datagen_rl_70B.py \
#     data.data_path=/nobackup/qinghao/runs/reasoning/Eurus_320_Llama3.3_70B \
#     data.save_dir="${SAVE_DIR}/Eurus-320-Llama-3.3-70B-Instruct" \
#     model.base_model_path=/nobackup/model/llama3.3/Llama-3.3-70B-Instruct \
#     data.process_rank=0 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python fastrl/trainer/eagle_datagen_rl_70B.py \
#     data.data_path=/nobackup/qinghao/runs/reasoning/Eurus_320_Qwen2.5_32B \
#     data.save_dir="${SAVE_DIR}/Eurus-320-Qwen2.5_32B" \
#     model.base_model_path=/nobackup/model/qwen2.5/Qwen2.5-32B \
#     data.process_rank=0 

