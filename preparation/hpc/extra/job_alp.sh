#!/bin/sh
​#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=alp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/alp%j.output
#SBATCH --mail-user=fb6372@partner.kit.edu
​
module load devel/cuda/11.8
​
source $HOME/miniconda3/etc/profile.d/conda.sh
​
python3 -u alpaca.py