`python report/parse_txts.py`

`python preparation/clean_abstracts.py analysis_works.csv --folder report `

`python preparation/extract_elements.py analysis_works.cleaned.works.csv --folder report/`

Copy `analysis_works.elements.works.csv` to BWUniCluster

Extract concepts with model `ft-xxl`:

```
#!/bin/sh
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --job-name=inf-analysis
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/inf%j.output
#SBATCH --mail-user=fb6372@partner.kit.edu
module load devel/cuda/11.8
source $HOME/miniconda3/etc/profile.d/conda.sh

python3 -u inference.py \
--input_file analysis_works.elements.works.csv \
--llama_variant 13B-v2 \
--model_id ft-xxl \
--batch_size 20 \
--inf_limit 60 \
--max_new_tokens 650
```

[...]

# Project Embeddings:

`python model/reduce_dim.py --input_file report/pdf/word-embs.full.M.pkl.gz report/pdf/transformed.pkl.gz``
