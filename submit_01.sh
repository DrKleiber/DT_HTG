#!/bin/bash

#SBATCH --job-name=DT_HTG
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# srun python ./train_MeanStd.py > test.log
# export PATH=/soft/swing/pytorch/1.10/cuda-11.3/bin:$PATH
srun python ./HTGNN_train.py > powerDrop_01.log
