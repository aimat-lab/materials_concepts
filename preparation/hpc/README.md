## Preparation with High Performance Computing (HPC)

- [x] Discuss the 5 annotated examples with Pascal
- [ ] Annote 100 abstracts with concepts: Progress 40/100
- [ ] Get access to weights directory (Felix)
- [ ] Fine-tune the model (Script)
- [ ] Generate concepts for 300 more abstracts
- [ ] Correct 300 concepts
- [ ] Fine-tune on these 300 abstracts

## Cuda Compatible Installation

For CUDA 10.2:
`pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102`

## Job.sh

```
#!/bin/sh
​#SBATCH --partition=gpu_4_a100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=gpu_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/fine_tune%j.output
#SBATCH --mail-user=fb6372@partner.kit.edu
​
module load devel/cuda/10.2
​
source $HOME/miniconda3/etc/profile.d/conda.sh
​
python3 -u fine-tune_model_trainer.py
```

## Convert Weights

`python3 conversion_script.py --input_dir ./LlamaW --model_size 13B --output_dir ./llama-13B`

## Commands

File size:
`du -shx *`

Go to Workspace:
`cd $(ws_find matconcepts)`

Available resources at the moment:
`sinfo_t_idle`

Copy files from $HOME to WS:
`cp $HOME/train.py .`
`cp $HOME/inference.py .`

Dispatch jobs:
`sbatch --partition=gpu_4_a100 job_train.sh`
`sbatch --partition=gpu_4_a100 job_inf.sh`

List current jobs:
`squeue`

Detailed information:
`scontrol show job <id>`
