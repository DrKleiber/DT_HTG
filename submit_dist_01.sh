#!/bin/bash

#SBATCH --job-name=DT_HTG
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --time=72:00:00

# srun python ./train_MeanStd.py > test.log
# export PATH=/soft/swing/pytorch/1.10/cuda-11.3/bin:$PATH
srun torchrun HTGNN_train_dist.py > powerDrop_dist_03.log
