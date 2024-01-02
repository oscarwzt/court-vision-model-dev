#!/bin/bash

#SBATCH --job-name=yolov8_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:2

module purge
module load anaconda3/2020.07
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zw2688/conda_env/yolonas;
export PATH=/scratch/zw2688/conda_env/yolonas/bin:$PATH;

cd /scratch/zw2688/Court_Vision_Model_Dev
python training_yolov8.py
