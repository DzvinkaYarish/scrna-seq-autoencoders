#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40g
#SBATCH -t 48:00:00

module load any/python/3.8.3-conda

conda activate scarches_p_3_10

python3.10 train_scpoli.py
