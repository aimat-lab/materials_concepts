# System Requirements

### Hardware requirements

This package requires only a standard computer with enough RAM to support the in-memory operations. Models were trained on a NVIDIA Tesla V100 from the [BWUniCluster 2.0](https://wiki.bwhpc.de/e/BwUniCluster2.0/Hardware_and_Architecture).

### Software requirements

#### OS Requirements

This package is supported for Linux. The package has been tested on the following systems:

- Linux: Red Hat Enterprise Linux release 9.4 (Plow)

#### Python Dependencies

This repo's code depends mainly on:

- `pandas`
- `nltk`
- `rake_nltk`
- `fire`
- `langdetect`
- `scipy`
- `networkx`
- `pyarrow`
- `tqdm`
- `jupyter`
- `torch`
- `sklearn`
- `transformers`

# Installation

Create a new virtual environment using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Install the local package in editable mode:

```bash
python -m pip install --no-deps --disable-pip-version-check -e .
```

Installation should take **5-10 min**.

# Reproducing paper results

Notes on how to reproduce the AUROC values present in the paper can be found in `Reproduce.md`.

# Dataset Creation

Note: You'll either need to download the figshare data or you can run the whole process on a small testset that we were able to store in this repo itself.

## Create data/ folder

Create `data/` top-level folder and a `table/` subfolder to store the dataset.

## Data Fetching

``$ python materials_concepts/dataset/downloader/download_sources.py --query 'materials science' --out data/table/test_3810.csv``

> This will create a `data/materials-science.sources.csv` file with all the sources.

Fetch works from single source:

``$ python materials_concepts/dataset/downloader/download_works.py fetchsingle --source S82336448 --out data/table/S82336448.works.csv``

> This will create a `S82336448.csv` file with all the works belonging to that source.

Fetch works from all sources:

``$ python materials_concepts/dataset/downloader/download_works.py fetchall --sources data/materials-science.sources.csv --out data/table/materials-science.works.csv``

> During fetching, this will create a `{source}.csv` file for each source in `cache` listing all the works which belong to that source. After downloading, these are merged automatically into a single file `out`.
> If the download gets interrupted, the downloaded files serve as a cache. Re-run the script, it will automatically skip sources for which the data was already fetched.

## Data Filtering

Filter the data to improve its quality:

``$ python materials_concepts/dataset/filtering/filter_works.py --source data/table/materials-science.works.csv --out data/table/materials-science.filtered.works.csv --njobs 8 --min-abstract-length 250 --max-abstract-length 3000 --topic "Materials science"``

> This will output a file `materials-science.filtered.works.csv` in the `data/table/` containing all works which sufficed the conditions.

## Data Preparation

In the end, the data folder should be structured like this. File names can be varied if the corresponding cli args to the individual scripts are adapted.

```bash
data
├── graph
│   └── edges.M.pkl
├── model
│   ├── baseline
│   │   ├── features.2016.binary.M.pkl.gz
│   │   ├── features.2019.binary.M.pkl.gz
│   │   ├── features.2022.binary.M.pkl.gz
│   │   └── model.pt
│   ├── combi
│   │   └── model.pt
│   ├── pure_embs
│   │   ├── features.concept-embs.2016.M.pkl.gz
│   │   ├── features.concept-embs.2019.M.pkl.gz
│   │   ├── features.concept-embs.2022.M.pkl.gz
│   │   └── model.pt
│   ├── test.data.M.pkl
│   └── train_val.M.pkl
└── table
    ├── lookup
    │   └── lookup.M.2.csv
    ├── materials-science.llama2.works.csv
    └── materials-science.sources.csv
```

### Cleaning abstracts

Clean the abstracts:

`$ python materials_concepts/dataset/preparation/clean_abstracts.py materials-science.filtered.works.csv --folder data/`

> This will output a file `materials-science.cleaned.works.csv` in the specified folder containing all works with cleaned abstracts.

## Data Enrichment

> Note: As these operations are very time consuming, the scripts make use of parallelization.

### Materials Extraction

Extract 'chemical elements' from abstracts:

`$ python materials_concepts/dataset/preparation/extract_elements.py materials-science.cleaned.works.csv --folder data/`

> This will output a file `materials-science.elements.works.csv` in the specified folder containing all works with extracted chemical elements in a separate columns `elements`.

### Concept Extraction (DEPRECATED)

Extract 'concepts' from abstracts using several methods (RAKE, keyBERT, OpenAlex concept list, Searching 'keywords' in abstracts):

`$ python materials_concepts/dataset/preparation/extract_concepts.py materials-science.elements.works.csv {method} {colname} --folder data/`

e.g.:

`$ python materials_concepts/dataset/preparation/extract_concepts.py materials-science.elements.works.csv rake rake_concepts --folder data/`

> This will output a file `materials-science.rake.works.csv` in the specified folder containing all works with extracted concepts according to `rake` (`{method}`) in a separate columns `rake_concepts` (`{colname}`).

### Concept Extraction (LLM)

The used concepts are generated by utilizing an LLM (LLaMa-2-13B) that is fine-tuned to this downstream task.

The processed file `materials-science.llama2.works.csv`, containing all the works can be downloaded from figshare.

# Classification

## Build Graph

Build concepts graph by executing the following command:

```
python materials_concepts/graph/build.py \
  --input_path data/table/materials-science.llama2.works.csv \
  --output_path data/graph/edges.M.pkl \
  --output_lookup_path data/table/lookup/lookup.M.csv \
  --colname llama_concepts \
  --min_occurence 3 \
  --min_words 2 \
  --max_words 20 \
  --min_occurence_elements 3 \
  --min_amount_elements 2

# for the small test set, this will yield:
# nodes: 4,078
# edges: 142,040
```

Produces a pickled file `graph/edges.pkl` containing the graph:

```
{
  "num_of_vertices": 123456,
  "edges": [(v1, v2, timestamp), (v1, v2, timestamp), ...],
}
```

Because of the sparse nature of the graph, it is stored as edge list. The timestamp is the number of days passed since `01-01-1970`.

> Note: If you want use rake concepts, you have to first extract the rake concepts and then replace `llama_concepts` with `rake_concepts` in the command above.

> Note: The concepts are run against a filter mechanism to remove concepts which are not relevant for the domain. The filters are stored in the same file and can be extended or modified as needed.

## Generate Raw Classification Task Data

Generate training and test data for classification task: Given {n} vertex pairs, decide whether they will be connected or not in {delta} years.

```
python materials_concepts/model/create_data.py \
 --graph_path data/graph/edges.M.pkl \
 --data_path data/model/data.pkl \
 --year_start_train 2016 \
 --year_start_test 2019 \
 --year_delta 3 \
 --edges_used_train 5_000_000 \ # 20_000 for small dataset
 --edges_used_test 2_000_000 \ # 2_000 for small dataset
 --train_val_split 0.8 \
 --min_links 1 \
 --max_v_degree=None \
 --verbose=True
```

Output:

```
{
  "year_train": 2016,
  "year_test": 2019,
  "year_delta": 3,
  "min_links": 1,
  "max_v_degree": None,
  "X_train": [(v1, v2), ...] unnconnected vertex pairs until 2016, (80%)
  "y_train": [0, 1, 1, 0, ...] indicating whether the vertex pairs will be connected in 2019 (2016 + 3) (80%)
  "X_val": (20%) unnconnected vertex pairs until 2016,
  "y_val": (20%) whether the vertex pairs will be connected in 2019,
  "X_test": [(v1, v2), ...] unnconnected vertex pairs until 2019,
  "y_test": whether the vertex pairs will be connected in 2022,
}
```

## Classification Process

The classification process can typically be divided into two steps:

1. Generate embeddings for nodes
2. Train a (binary) classifier on the (concatenated) embeddings

### Baseline Model

1. Generate the embeddings

Embeddings for training:

```
python -u materials_concepts/model/combi/pre_compute.py \
  --graph_path data/graph/edges.M.pkl \
  --output_path data/model/baseline/features.2016.binary.M.pkl.gz \
  --binary True \
  --years "[2012, 2013, 2014, 2015, 2016]"
```

Embeddings for validation:

```
python -u materials_concepts/model/combi/pre_compute.py \
  --graph_path data/graph/edges.M.pkl \
  --output_path data/model/baseline/features.2019.binary.M.pkl.gz \
  --binary True \
  --years "[2015, 2016, 2017, 2018, 2019]"
```

2. Train the model

```
python -u materials_concepts/model/combi/train.py \
  --data_path data/model/data.pkl \
  --emb_f_train_path data/model/baseline/features.2016.binary.M.pkl.gz \
  --emb_f_test_path data/model/baseline/features.2019.binary.M.pkl.gz \
  --emb_c_train_path False \
  --emb_c_test_path False \
  --lr 0.0005 \
  --batch_size 1000 \
  --num_epochs 10000 \
  --pos_ratio 0.3 \
  --layers "[20, 300, 180, 108, 64, 10, 1]" \
  --step_size 200 \
  --gamma 0.95 \
  --dropout 0.1 \
  --sliding_window 5 \
  --log_interval 200 \
  --log_file logs/07_train-baseline.log \
  --save_model "data/model/baseline/model.pt"
```

### Pure Embeddings Model

0. Generate the emebddings (see `Word Embeddings` below)
1. Train the model

```
python -u materials_concepts/model/combi/train.py \
  --data_path data/model/data.pkl \
  --emb_f_train_path ""  \
  --emb_f_test_path "" \
  --emb_c_train_path data/model/combi/word-embs.2016.M.pkl.gz \
  --emb_c_test_path data/model/combi/word-embs.2019.M.pkl.gz \
  --lr 0.001 \
  --batch_size 1000 \
  --num_epochs 15000 \
  --pos_ratio 0.3 \
  --layers "[1536, 1024, 819, 10, 1]" \
  --step_size 200 \
  --gamma 0.9 \
  --dropout 0.1 \
  --sliding_window 5 \
  --log_interval 200 \
  --log_file logs/07_train-wembs.log \
  --save_model "data/model/pure_embs/model.pt"
```

### Combination of features

1. Use concatentation of baseline features and word embeddings as input. Take a look at the chapter `Word Embeddings` to see how to generate word embeddings.
2. Train the model

```
python -u model/combi/train.py \
  --data_path data/model/data.M.pkl \
  --emb_f_train_path data/model/baseline/features.2016.binary.M.pkl.gz \
  --emb_f_test_path data/model/baseline/features.2019.binary.M.pkl.gz \
  --emb_c_train_path data/model/combi/word-embs.2016.M.pkl.gz \
  --emb_c_test_path data/model/combi/word-embs.2019.M.pkl.gz \
  --lr 0.001 \
  --batch_size 1000 \
  --num_epochs 15000 \
  --pos_ratio 0.3 \
  --layers "[1556, 1556, 933, 559, 335, 10, 1]" \
  --step_size 200 \
  --gamma 0.95 \
  --dropout 0.1 \
  --sliding_window 5 \
  --log_interval 200 \
  --log_file logs/06_train-combi.log \
  --save_model "data/model/combi/model.pt"
,

```

### Combination of models

No need to train anything, as we just combine a baseline with a pure embeddings model.

# Evaluation

To evaluate the models and reproduce the results, take a look at `Reproduce.md`.

# Word Embeddings

## Generate Word Embeddings

Word embeddings are generated using BERT or a fine-tuned version of BERT e.g. MatSciBERT.
To extract ambeddings for all concepts (all embedded tokens comprising a concept are `averaged`), run:

```
python -u materials_concepts/word_embeddings/generate.py \
  --concepts_path data/table/materials-science.llama2.works.csv \
  --lookup_path data/table/lookup/lookup.M.csv \
  --output_path data/embeddings/ \
  --log_to_stdout False \
  --step_size 50 \
  --start 0 \
  --end 5000
```

> Currently, if a concept is not exactly contained in the abstract (this can happen because LLMs can apply some "normalization" during extraction), the embedding vector is set to the average of all tokens in the abstract.
> On GPU4_A100 generating embeddings for 80k abstracts takes about 6h.

## Average Word Embeddings

Averaging word (concept) embeddings so that they can be used as feature vectors for classification.

```
python materials_concepts/word_embeddings/average_embs.py \
  --concepts_path data/table/materials-science.llama2.works.csv \
  --lookup_path data/table/lookup/lookup.M.csv \
  --filter_path data/table/lookup/lookup.M.csv \
  --embeddings_dir data/embeddings/ \
  --output_path data/model/combi/word-embs.2016.M.pkl.gz \
  --store_concepts_plain False \
  --until_year 2016
```

# Interview

## LLM Report

Generate `distilled` version of reports:

```bash
python materials_concepts/report/pdf/generation/hack_llm_ready_report.py
```

Generate the LLM report (prompt engineering + some report sections => LLM APIs)
from the "distilled" version of the reports:

```bash
export RESEARCHER="...";

python materials_concepts/report/generate_llm_selection.py --txt_path materials_concepts/report/prompt_sec3.txt --tex_path materials_concepts/report/pdf/generation/${RESEARCHER}/distilled/plain_suggestions.tex --output_path materials_concepts/report/pdf/generation/${RESEARCHER}/llm_report_sec3.txt

python materials_concepts/report/generate_llm_selection.py --txt_path materials_concepts/report/prompt_sec5.txt --tex_path materials_concepts/report/pdf/generation/${RESEARCHER}/distilled/exotic_suggestions.tex --output_path materials_concepts/report/pdf/generation/${RESEARCHER}/llm_report_sec5.txt
```
