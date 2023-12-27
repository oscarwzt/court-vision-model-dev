#!/bin/bash

# Parameters for job submission
time=${1:-"05:00:00"}   # Default time if not provided
mem=${2:-"32GB"}        # Default memory if not provided
gres=${3:-"gpu:2"}      # Default GPU resources if not provided
cpus=${4:-"8"}          # Default CPUs if not provided

# Submit the job and extract job ID
job_output=$(/opt/slurm/bin/sbatch --time=$time --mem=$mem --wrap="sleep infinity" --gres=$gres --cpus-per-task=$cpus)
job_id=$(echo $job_output | awk '{print $NF}')

# Initialize variable for compute node
compute_node=""

# Poll for the compute node assignment
while [ -z "$compute_node" ]; do

    # Fetch job status
    job_status=$(/opt/slurm/bin/squeue --job $job_id | grep $job_id)

    # Extract the job state and compute node
    job_state=$(echo "$job_status" | awk '{print $5}')
    potential_node=$(echo "$job_status" | awk '{print $NF}')

    # Check if the job state is running or if a node name is present
    if [ "$job_state" == "R" ] || [[ "$potential_node" =~ ^[a-zA-Z]+[0-9]+$ ]]; then
        compute_node=$potential_node
    else
        echo "Job $job_id is still pending or not assigned to a node..."
        sleep 5  # Wait for 5 seconds before checking again
    fi
done

echo $compute_node  # Output only the compute node name
