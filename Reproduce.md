To reproduce the reported AUC values, make sure to download all the figshare data and arange it in a `test-data` dir (or anywhere else) like this:

```bash
test-data/
├── baseline
│   ├── auc_curve.pkl.gz
│   ├── features.2016.binary.M.pkl.gz
│   ├── features.2019.binary.M.pkl.gz # <- this will be used for eval
│   ├── features.2022.binary.M.pkl.gz # <- this is only for predictions with all model knowledge available
│   └── model.pt
├── combi
│   └── model.pt
├── mixture # <- not really needed, since it's only a blend of models
├── pure_embs
│   ├── features.concept-embs.2016.M.pkl.gz
│   ├── features.concept-embs.2019.M.pkl.gz # <- this will be used for eval
│   ├── features.concept-embs.2022.M.pkl.gz # <- this is only for predictions with all model knowledge available
│   └── model.pt
└── test.data.M.pkl
```

Run the `reproduce.sh` script that will evaluate the models on the test data. You'll need to adapt your storage layout to the script or vice versa, since figshare only allows for uploading a flattened directory.

Before running, please make sure you've followed the installation guide and your virtual env is activated. Also make sure to run the script on a machine with sufficient RAM. The inference can be carried out on a CPU, but 2 million data points to evaluate are quite many. Therefore, we emply a batched inference with a `chunk_size` of 10k, feel free to adapt that in the reproduce script. 

The metrics and predictions are saved, allowing to calculate the ROC curves as well using `sklearn.metrics.roc_curve`.

## Further Reproducibility Notes

Edge construction thresholds are, as mentioned in the main README:

```
  [...]
  --min_occurence 3 \
  --min_words 3 \
  --max_words 20 \
  --min_occurence_elements 3 \
  --min_amount_elements 2
```

MatSciBERT from hugging face (`m3rg-iitd/matscibert`) with commit hash: `24a4e4318dda9bc18bff5e6a45debdcb3e1780e3`.

BERT from hugging face (`bert-base-uncased`) with commit hash: `86b5e0934494bd15c9632b12f734a8a67f723594`.

Random seed for all numpy and torch operations: `42` (this is also the default setting).

## OpenAlex Data Collection

The dataset used in this work was collected from the [OpenAlex API](https://openalex.org/) using a two-step query process:

1. **Source Discovery**: We queried OpenAlex sources (host venues) using the search term `"materials science"` to identify relevant journals and venues in the materials science domain. This retrieves sources like journals, conferences, and repositories that publish materials science content.

2. **Works Retrieval**: For each identified source, we fetched all associated works (publications) including their abstracts, concepts, publication dates, and metadata.

The specific fields retrieved for each work include: `id`, `doi`, `display_name`, `publication_date`, `is_retracted`, `is_paratext`, `abstract_inverted_index`, and `concepts`.

**Important Note on Data Reproducibility**: Instead of pinning to a specific OpenAlex snapshot (which can be difficult to maintain and access), we provide the raw data dump resulting from our query on Figshare. This ensures exact reproducibility of our results while making the data easily accessible.

The data collection scripts can be found in:
- `materials_concepts/dataset/downloader/download_sources.py` - for retrieving sources
- `materials_concepts/dataset/downloader/download_works.py` - for retrieving works from each source

## Evaluating a model on Mario Krenn's Science4Cast challenge

All information regarding the challenge can be found [here](https://github.com/artificial-scientist-lab/FutureOfAIviaAI?tab=readme-ov-file). The necessary data can be 

The data structure is slightly different, so some adjustments are necessary:

- Origin DAY needs to be adjusted to `1990-01-01` in `materials_concepts/utils/constants.py`
- The data format in the `.pkl` files needs to be ported slightly
  - Note: the data can be found [here](https://zenodo.org/records/7882892#.ZE-Egx9BwuU) 
  - The portation can be done in a simple Python interpreter with the code shown below
  - `all_edges.pkl`: All edges of the final graph, we'll use that to create a graph up until 2014 and 2017 for training and validation.
  - `SemanticGraph_delta_3_cutoff_0_minedge_1.pkl`: Contains the test edges and their labels used in the challenge

```python
import pickle
import numpy as np

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

all_edges = load_pickle('all_edges.pkl')
write_pickle({"edges": np.array(all_edges)}, "ported_all_edges.pkl")

ground_truth = load_pickle('SemanticGraph_delta_3_cutoff_0_minedge_1.pkl')
write_pickle({"X_test": ground_truth[1], "y_test": ground_truth[2]}, "ported_SemanticGraph_delta_3_cutoff_0_minedge_1.pkl")
```