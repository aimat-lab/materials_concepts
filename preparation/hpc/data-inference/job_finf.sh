#!/bin/sh
#SBATCH --partition=gpu_4_a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=finf-11000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=finf/finf-11000.output
#SBATCH --mail-user=fb6372@partner.kit.edu

module load devel/cuda/11.8
​
source $HOME/miniconda3/etc/profile.d/conda.sh
​
python3 -u ../full_inference.py \
 --llama_variant 13B \
 --model_id full-finetune \
 --start 11000 \
 --n 3000 \
 --input_file data/works.csv \
 --batch_size 25 \
 --max_new_tokens 512 \
 --cwd ../
