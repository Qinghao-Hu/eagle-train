#!/bin/bash -l
#SBATCH --partition="backfill,high_prio,batch"
#SBATCH --account=nvr_elm_llm
#SBATCH -J eagle-mix
#SBATCH -N 12
#SBATCH --exclusive
#SBATCH -o slurm/%x-%j.out                 # output file (%j expands to jobID)
#SBATCH -e slurm/%x-%j.err                  # error log file (%j expands to jobID)
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH -c 32
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --time=0-04:00:00                           # Time limit (hh:mm:ss)

# !!!! Load your own environment here !!!! #
# !!!! Load your own environment here !!!! #
set -e
source ~/.bashrc
source /home/jerguo/miniconda3/bin/activate eagle

MODEL_NAME=Llama-3.1-8B-Instruct
BASE_MODEL_PATH=/nobackup/model/llama3.1/${MODEL_NAME}
DATA_PATH=/home/jerguo/dataset/datasets/tlt/eagle-mix
SAVE_DIR=/home/jerguo/dataset/datasets/tlt/eagle-processed/Eagle-Mix-${MODEL_NAME}

run_name="$(basename $MODEL_NAME)_eagle-mix"
out_dir="logs/${run_name}"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_gpus=${NUM_GPUS:-$num_gpus}

num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
if [ $num_nodes == 0 ]; then
    num_nodes=1
fi
num_nodes=${NUM_NODES:-$num_nodes}



if [ $num_nodes -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_addr=${MASTER_ADDR:-$master_addr}

    # Launch via srun
    header="srun torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$master_addr:56321 \
    --nnodes=$num_nodes \
    --nproc-per-node=$num_gpus \
    -m eagle_datagen.py"
else
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
 
    # Launch without srun
    header="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m eagle_datagen.py"
fi
echo "slurm_nodelist=${SLURM_NODELIST} num_nodes=${num_nodes} master_addr=${master_addr} master_port=${master_port} num_gpus=${num_gpus}"

export OMP_NUM_THREADS=$num_gpus

base_arguments=(
    --model.base_model_path $BASE_MODEL_PATH
    --data.data_path $DATA_PATH
    --data.save_dir $SAVE_DIR
    --data.max_length 4096
)

base_arguments+=( $@ )

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out
