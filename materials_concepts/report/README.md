## 1

Put abstracts as separate .txt files in `report/abstracts/` folder.
`python report/parse_txts.py`

## 2

`python preparation/clean_abstracts.py --input report/analysis_works.csv --output report/analysis_works.cleaned.works.csv`

## 3

`python preparation/extract_elements.py --input report/analysis_works.cleaned.works.csv --output report/analysis_works.elements.works.csv`

Copy `analysis_works.elements.works.csv` to BWUniCluster:

```
scp report/analysis_works.elements.works.csv {CLUSTER_USER}@bwunicluster.scc.kit.edu:/pfs/work7/workspace/scratch/{CLUSTER_USER}-matconcepts/pfs/work7/workspace/scratch/{CLUSTER_USER}-matconcepts/
```

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
#SBATCH --mail-user={CLUSTER_USER}@partner.kit.edu
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

Adjust --inf_limit and time accordingly. Run `sbatch infere.sh`. `analysis_works.elements.works.csv` needs to contain column `id` and `abstract`.

Time estimation: 300 abstracts with batch_size 20 takes ~730 seconds (12 minutes).

Copy back to local machine:

```
scp {CLUSTER_USER}@bwunicluster.scc.kit.edu:/pfs/work7/workspace/scratch/{CLUSTER_USER}-matconcepts/pfs/work7/workspace/scratch/{CLUSTER_USER}-matconcepts/data/inference_13B-v2/ft-xxl/2024-01-24_23-16-11_ft-xxl.csv report/concepts.csv
```

## Post-proces concepts

`python report/post_process_concepts.py --concepts report/concepts.csv --to_merge report/analysis_works.elements.works.csv --output report/fixed_concepts.csv`

# Run combinations inference

1. Upload `fixed_concepts.csv` to BWUniCluster

   ```
   scp report/fixed_concepts.csv {CLUSTER_USER}@bwunicluster.scc.kit.edu:/pfs/work7/workspace/scratch/{CLUSTER_USER}-linkpr/{CLUSTER_USER}-graph/backend/predictions/
   ```

2. Run `job.sh` in `backend` folder which will run `analysis_inference.py`.
   Make sure to adapt the --concepts argument in `job.sh` to the correct path.

3. Download `.pkl` files with predictions to `report/pdf/generation/` folder.
   `scp {CLUSTER_USER}@bwunicluster.scc.kit.edu:/pfs/work7/workspace/scratch/{CLUSTER_USER}-linkpr/{CLUSTER_USER}-graph/backend/predictions/researcher.txt.pkl report/pdf/generation/predictions/researcher.pkl`

## Generate PDFs

Place `fixed_concepts.csv` in `report/pdf/generation` folder.

Edit `report/pdf/advanced_analysis.py` to list `sources` that should be processed.

## Project Embeddings:

(You only need to run this once!)

Download embeddings `word-embs.full.M.pkl.gz` from BWUniCluster.

Use `reduce_dim.py` to reduce the dimensionality of the embeddings to 2D:
`python model/reduce_dim.py --input_file report/pdf/generation/word-embs.full.M.pkl.gz report/pdf/generation/transformed.pkl.gz``

# TODO

- [x] Update this README
- [ ] Workflow -> Generate Reports
- [ ] Map: own concepts highlighted and suggested concepts highlighted
- [x] Potentially interesting concepts: Node_Degree <= N && min_dist <= Concepts_Distance <= max_dist
- [ ] Works: request unique OpenAlex WorkID
- [x] Evolution: average for all years and month = 0
- [ ] "type heterojunction" is missing from embeddings because of one work (OpenAlex ID) exists 8 times -> Clean up
- [ ] Concept combination prediction step should write files with `.txt` part to disc
