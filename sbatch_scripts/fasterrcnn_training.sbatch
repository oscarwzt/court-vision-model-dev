#!/bin/bash

#SBATCH --job-name=faster_rcnn_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:4

module purge
module load anaconda3/2020.07
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/zw2688/conda_env/yolonas;
export PATH=/scratch/zw2688/conda_env/yolonas/bin:$PATH;
export MASTER_ADDR='localhost' 

cd /scratch/zw2688/DL_project/fastercnn-pytorch-training-pipeline
export MASTER_ADDR='localhost' 
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py --data data_configs/basketball_voc.yaml --epochs 250 --lr 0.0001 --model fasterrcnn_resnet50_fpn --use-train-aug --name basketball_detect_training_batch64_2 --batch 16 --workers 12 
