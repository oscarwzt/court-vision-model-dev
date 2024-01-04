#!/bin/bash

#SBATCH --job-name=cls_hyper_tune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:3

module purge
module load anaconda3/2020.07
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zw2688/conda_env/yolonas;
export PATH=/scratch/zw2688/conda_env/yolonas/bin:$PATH;

cd /scratch/zw2688/Court_Vision_Model_Dev
export CUDA_VISIBLE_DEVICES=0,1,2
python hyper_param_tuning.py
