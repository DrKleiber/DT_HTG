#!/bin/bash

#SBATCH --job-name=DT_HTG_2parts
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00

# srun python ./train_MeanStd.py > test.log
# export PATH=/soft/swing/pytorch/1.10/cuda-11.3/bin:$PATH
srun torchrun HTGNN_train_dist_2parts.py > powerDrop_dist_2parts_test.log
