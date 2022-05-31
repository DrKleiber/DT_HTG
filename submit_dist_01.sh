#!/bin/bash

#SBATCH --job-name=DT_HTG
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

# srun python ./train_MeanStd.py > test.log
# export PATH=/soft/swing/pytorch/1.10/cuda-11.3/bin:$PATH
srun python -m torch.distributed.run --rdzv_backend=c10d --nnodes=1 --nproc_per_node=2 ./HTGNN_train_dist.py > powerDrop_dist_test.log
