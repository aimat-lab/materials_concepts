## Preparation with High Performance Computing (HPC)

- [x] Discuss the 5 annotated examples with Pascal
- [ ] Annote 100 abstracts with concepts: Progress 40/100
- [ ] Get access to weights directory (Felix)
- [ ] Fine-tune the model (Script)
- [ ] Generate concepts for 300 more abstracts
- [ ] Correct 300 concepts
- [ ] Fine-tune on these 300 abstracts

## Cuda Compatible Installation

For CUDA 11.8:
`pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118`

## Workspace: Matconcepts

`/pfs/work7/workspace/scratch/fb6372-matconcepts`

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
