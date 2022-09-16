#!/bin/bash

#SBATCH --job-name=DT_gFHR
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --time=72:00:00

# srun python ./train_MeanStd.py > test.log
# export PATH=/soft/swing/pytorch/1.10/cuda-11.3/bin:$PATH
srun torchrun gFHR_train_dist_2parts.py > gFHR_dist_2parts_test.log
