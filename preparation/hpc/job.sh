#!/bin/sh
​#SBATCH --partition=dev_single
#SBATCH --time=06:00:00
#SBATCH --job-name=fine_tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/fine_tune%j.output
#SBATCH --error=logs/fine_tune%j.error
#SBATCH --mail-user=fb6372@partner.kit.edu
​
module load devel/cuda/10.2
​
source $HOME/miniconda3/etc/profile.d/conda.sh
​
python3 -u fine-tune_model.py