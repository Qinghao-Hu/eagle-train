#!/bin/bash
#SBATCH -A nvr_elm_llm                  #account
#SBATCH -p batch                        #partition
#SBATCH -t 04:00:00                     #wall time limit, hr:min:sec
#SBATCH -N 16                           #number of nodes
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

BASE_MODEL_PATH=${1}
cmd="/home/jerguo/projects/tlt-workspace/eagle-train/scripts/launch_by_slurm/train_eagle3.sh $BASE_MODEL_PATH /home/jerguo/dataset/datasets/tlt"

srun \
    -o $OUTFILE -e $ERRFILE \
    bash -c "${cmd}"