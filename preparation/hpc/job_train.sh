#!/bin/sh
​#SBATCH --partition=gpu_4_a100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/train%j.output
#SBATCH --mail-user=fb6372@partner.kit.edu
​
module load devel/cuda/11.8
​
source $HOME/miniconda3/etc/profile.d/conda.sh
​
python3 -u fine-tune_model_trainer.py