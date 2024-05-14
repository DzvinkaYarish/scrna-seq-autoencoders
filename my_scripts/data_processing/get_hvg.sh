#!/bin/bash

#SBATCH -J get_hvg
#SBATCH --partition=main
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH -t 2:00:00

module load any/python/3.8.3-conda

conda activate scarches_p_3_10

python3.10 get_hvg.py