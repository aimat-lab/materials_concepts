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

## Post-proces concepts

`python report/post_process_concepts.py --concepts report/concepts.csv --to_merge report/analysis_works.elements.works.csv --output report/fixed_concepts.csv`

# Run combinations inference

1. Upload `fixed_concepts.csv` to BWUniCluster
2. Run `job.sh` in `backend` folder which will run `analysis_inference.py`.
3. Download `.pkl` files with predictions to `report/pdf/generation/` folder.
   `scp fb6372@bwunicluster.scc.kit.edu:/pfs/work7/workspace/scratch/fb6372-linkpr/fb6372-graph/backend/predictions/christoph_brabec_all.txt.pkl report/pdf/generation/predictions/christoph_brabec.pkl`

## Project Embeddings:

Download embeddings `word-embs.full.M.pkl.gz` from BWUniCluster.

Use `reduce_dim.py` to reduce the dimensionality of the embeddings to 2D:
`python model/reduce_dim.py --input_file report/pdf/generation/word-embs.full.M.pkl.gz report/pdf/generation/transformed.pkl.gz``

# TODO

- [ ] Update this README
- [ ] Workflow -> Generate Reports
- [ ] Map: own concepts highlighted and suggested concepts highlighted
- [ ] Potentially interesting concepts: Node_Degree <= N && min_dist <= Concepts_Distance <= max_dist
- [ ] Works: request unique OpenAlex WorkID
- [x] Evolution: average for all years and month = 0
- [ ] "type heterojunction" is missing from embeddings because of one work (OpenAlex ID) exists 8 times -> Clean up
