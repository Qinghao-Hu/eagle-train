#!/bin/bash
#SBATCH -A nvr_elm_llm                  #account
#SBATCH -p "backfill,high_prio,batch,batch_short"                        #partition
#SBATCH -t 02:00:00                     #wall time limit, hr:min:sec
#SBATCH -N 4                            #number of nodes
#SBATCH --mem=0                         # all mem avail
#SBATCH --gres=gpu:8
#SBATCH -J nvr_elm_llm-tlt-eagle        #job name
#SBATCH --exclusive #important: need to make it exclusive to improve speed


WORKDIR=$(pwd)
TIME=$(date +"%m-%d_%H:%M")
RESULTS="${WORKDIR}/slurm-logs/$TIME"
OUTFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.out"
ERRFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.err"
mkdir -p ${RESULTS}
PROGRESS="$SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
HNAME=$(echo $(scontrol show hostname))


echo "[$HNAME] | $PROGRESS | $WORKDIR | Running Jobs"

BASE_MODEL_PATH=${1} #/home/jerguo/dataset/llama3.1/Llama-3.1-8B-Instruct
cmd="/home/jerguo/projects/tlt-workspace/eagle-train/scripts/launch_by_slurm/datagen.sh $BASE_MODEL_PATH /home/jerguo/dataset/datasets/tlt"

srun \
    -o $OUTFILE -e $ERRFILE \
    bash -c "${cmd}"