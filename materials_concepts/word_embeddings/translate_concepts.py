from ast import literal_eval
import pandas as pd

import logging
import os
from tqdm import tqdm
import pickle
import gzip
import sys


def legacy_prepare_dataframe(df, lookup_df, cols):  # TODO: Refactor?
    lookup = {key: True for key in lookup_df["concept"]}

    df.abstract = df.abstract.str.lower()
    df.llama_concepts = df.llama_concepts.apply(literal_eval).apply(
        lambda x: list({c.lower() for c in x if lookup.get(c.lower())})
    )

    df.elements = df.elements.apply(
        lambda str: list({e for e in str.split(",") if lookup.get(e)})
        if not pd.isna(str)
        else []
    )

    el_lookup = {(id, e.lower()): e for id, es in zip(df.id, df.elements) for e in es}

    df.elements = df.elements.apply(lambda x: [e.lower() for e in x])

    df.publication_date = pd.to_datetime(df.publication_date)

    df.concepts = df.llama_concepts + df.elements
    df.concepts = df.concepts.apply(lambda x: sorted(x))  # sort

    df["new_concepts"] = df.apply(
        lambda row: [el_lookup.get((row.id, c), c) for c in row.concepts], axis=1
    )

    return df[cols]


def setup_logger(level=logging.INFO, log_to_stdout=True):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    if log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler("logs.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


def save_compressed(obj, path):
    compressed = gzip.compress(pickle.dumps(obj))
    with open(path, "wb") as f:
        f.write(compressed)


def process_works(df, desc, store):
    for id, new_concepts in tqdm(zip(df.id, df.new_concepts), total=len(df), desc=desc):
        embeddings = store[id]

        store[id] = {
            concept: embedding for concept, embedding in zip(new_concepts, embeddings)
        }

    return store


concepts_path = "data/table/materials-science.llama.works.csv"
lookup_path = "data/table/lookup/lookup_large.csv"
input_path = "data/embeddings/large_fix/"
output_path = "data/embeddings/final_fix_v2/"

logger = setup_logger(level=logging.INFO, log_to_stdout=True)


df = legacy_prepare_dataframe(
    df=pd.read_csv(concepts_path),
    lookup_df=pd.read_csv(lookup_path),
    cols=["id", "concepts", "new_concepts"],
)


start = 0
end = 222_000
step_size = 500

for i in range(start, end, step_size):
    logger.info(f"Process {i} to {i+step_size}...")

    input_file = os.path.join(input_path, f"embeddings_{i:06d}.pkl.gz")
    save_file = os.path.join(output_path, f"embeddings_{i:06d}.pkl.gz")

    store = load_compressed(input_file)
    partial_df = df[i : i + step_size]
    store = process_works(partial_df, desc=f"Generate embeddings ({i})", store=store)
    logger.info("Save embeddings")
    save_compressed(store, save_file)
