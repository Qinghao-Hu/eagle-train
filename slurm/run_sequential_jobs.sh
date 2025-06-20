#!/bin/bash

# Sequential training script for running train_512K.sh multiple times
# Each run uses internal checkpointing/restore functionality
# Jobs continue sequentially even if previous job times out or fails

set -e

# Configuration
NUM_RUNS=${NUM_RUNS:-5}
SCRIPT_PATH=${SCRIPT_PATH:-"./train_512K.sh"}

# Check if train_512K.sh exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: $SCRIPT_PATH not found!"
    echo "Usage: NUM_RUNS=3 SCRIPT_PATH=./my_script.sh $0"
    exit 1
fi

# Array to store job IDs
declare -a JOB_IDS=()

echo "Starting sequential training with $NUM_RUNS runs..."
echo "Script: $SCRIPT_PATH"
echo "Note: Using internal checkpointing - each run continues from previous checkpoint"
echo "Note: Jobs will continue even if previous job times out or fails"
echo "==========================================="

# Submit all jobs with dependencies
for ((i=1; i<=NUM_RUNS; i++)); do
    echo "Submitting run $i/$NUM_RUNS..."
    
    if [ $i -eq 1 ]; then
        # First job - no dependency
        JOB_OUTPUT=$(sbatch "$SCRIPT_PATH")
    else
        # Subsequent jobs depend on previous job completion (regardless of exit status)
        JOB_OUTPUT=$(sbatch --dependency=afterany:${JOB_IDS[-1]} "$SCRIPT_PATH")
    fi
    
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+$')
    JOB_IDS+=("$JOB_ID")
    
    if [ $i -eq 1 ]; then
        echo "Job $i submitted with ID: $JOB_ID (initial run)"
    else
        echo "Job $i submitted with ID: $JOB_ID (will start after ${JOB_IDS[-2]} completes)"
    fi
done

echo ""
echo "==========================================="
echo "All $NUM_RUNS jobs submitted successfully!"
echo ""
echo "Job dependency chain:"
for ((i=0; i<${#JOB_IDS[@]}; i++)); do
    if [ $i -eq 0 ]; then
        echo "  Run $((i+1)): ${JOB_IDS[i]} (initial run)"
    else
        echo "  Run $((i+1)): ${JOB_IDS[i]} (starts after run $i, regardless of exit status)"
    fi
done

echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor jobs: squeue -u \$USER"
echo "Cancel all jobs: scancel ${JOB_IDS[*]}"
echo "Check progress: tail -f slurm/train_512K-<job_id>.out"