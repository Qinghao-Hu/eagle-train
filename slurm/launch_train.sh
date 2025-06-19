#!/bin/bash
#SBATCH -A nvr_elm_llm                  #account
#SBATCH -p batch_block1,interactive                 #partition
#SBATCH -t 04:00:00                     #wall time limit, hr:min:sec
#SBATCH -N 2                           #number of nodes
#SBATCH --mem=0                         # all mem avail
#SBATCH --gres=gpu:8
#SBATCH -J nvr_elm_llm-tlt-eagle        #job name
#SBATCH --array=1-2%1
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

cmd="/home/shangy/TLT/eagle-train/scripts/launch_by_slurm/datagen.sh /home/shangy/TLT/models/Llama-3.1-8B-Instruct /home/shangy/TLT/dataset"

srun \
    -o $OUTFILE -e $ERRFILE \
    bash -c "${cmd}"