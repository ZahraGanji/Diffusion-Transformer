#!/bin/bash
#
#SBATCH --account holistic-vid-westai
#SBATCH --nodes=1
#SBATCH --job-name=DIT_Sample
#SBATCH --output=./out/train_net-%j.out
#SBATCH --error=./out/train_net-%j.err
#SBATCH --gres gpu:4
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=48
#SBATCH --partition=booster

module load Stages/2023
module load Python/3.11.3
module load CUDA/12
source /p/project/holistic-vid-westai/ganji1/pytorch_env/pytorch_new/bin/activate


echo "\n starting torchrun"

srun torchrun --standalone --nproc_per_node 4 train.py --model DiT-S/2 --data-path /p/scratch/holistic-vid-westai/veeramacheneni2_scratch/CelebAMask-HQ/data256x256