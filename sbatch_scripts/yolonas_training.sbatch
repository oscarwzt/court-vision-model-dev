#!/bin/bash

#SBATCH --job-name=yolonas_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:3

module purge
module load anaconda3/2020.07
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zw2688/conda_env/yolonas;
export PATH=/scratch/zw2688/conda_env/yolonas/bin:$PATH;

cd /scratch/zw2688/DL_project
python training_yolonas.py
