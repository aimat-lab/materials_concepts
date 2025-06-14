import gzip
import logging
import os
import pickle
import sys

import fire
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from materials_concepts.utils.utils import prepare_dataframe

tqdm.pandas()

DIM_EMBEDDING = 768
MAX_TOKENS = 510  # 512 - 2 (CLS and SEP)
CLS_TOKEN_ID = 102
SEP_TOKEN_ID = 103


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


def setup_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return tokenizer, model


def wrap(tokens):
    """Wrap tokens with [CLS] and [SEP], only if not already wrapped"""
    if tokens[0] != CLS_TOKEN_ID:
        tokens = [CLS_TOKEN_ID] + tokens

    if tokens[-1] != SEP_TOKEN_ID:
        tokens = tokens + [SEP_TOKEN_ID]

    return tokens


def init_get_embeddings(model, tokenizer):
    def func(text):
        tokens = tokenizer(text)["input_ids"]

        chunks = [
            wrap(tokens[i : i + MAX_TOKENS]) for i in range(0, len(tokens), MAX_TOKENS)
        ]

        embedded_chunks = []
        with torch.no_grad():
            for chunk in chunks:
                outputs = model(torch.tensor([chunk]))
                embeddings = outputs.last_hidden_state.squeeze()
                embedded_chunks.append(embeddings[1:-1])  # remove [CLS] and [SEP]

        return torch.cat(embedded_chunks)

    return func


def find_sequence_in_list(lst, seq):
    seq_len = len(seq)
    indices = []
    for i in range(len(lst)):
        if lst[i : i + seq_len] == seq:
            indices.append(i)
    return indices


def init_get_token_ids(tokenizer):
    def func(text):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        return token_ids

    return func


def get_concept_embedding(
    abstract_embedding, concept_tokens, indices, aggregation=torch.mean
):
    concept_embeddings = [
        abstract_embedding[i : i + len(concept_tokens)] for i in indices
    ]

    averaged_concept_embeddings = torch.stack(
        [aggregation(cs, dim=0) for cs in concept_embeddings]
    )

    return aggregation(averaged_concept_embeddings, dim=0)


def extract_embeddings_for_abstract(
    abstract_embedding, abstract_tokens, concepts, aggregation=torch.mean
):
    avg_embedding = torch.mean(abstract_embedding, dim=0)

    concept_embeddings = {}
    for concept in concepts:
        concept_tokens = get_token_ids(concept)
        indices = find_sequence_in_list(abstract_tokens, concept_tokens)
        logger.debug(f"Concept: {concept} - Occurences: {len(indices)}")

        concept_embedding = (
            get_concept_embedding(
                abstract_embedding, concept_tokens, indices, aggregation
            )
            if indices
            else avg_embedding  # use average embedding of abstract
        )

        concept_embeddings[concept] = concept_embedding

    return concept_embeddings


def save_compressed(obj, path):
    compressed = gzip.compress(pickle.dumps(obj))
    with open(path, "wb") as f:
        f.write(compressed)


def process_works(df, desc):
    store = {}

    for id, abstract, concepts in tqdm(
        zip(df.id, df.abstract, df.concepts, strict=False), total=len(df), desc=desc
    ):
        logger.info(
            f"Process {id}: abstract len of {len(abstract)} with {len(concepts)} concepts"
        )

        abstract_tokens = get_token_ids(abstract)

        abstract_embedding = get_embeddings(abstract)

        logger.debug(
            f"Tokenized abstract length: {len(abstract_tokens)}, Abstract embedding shape: {abstract_embedding.shape}"
        )

        embeddings = extract_embeddings_for_abstract(
            abstract_embedding, abstract_tokens, concepts, aggregation=torch.mean
        )

        assert len(embeddings.values()) == len(concepts)

        store[id] = embeddings

    return store


def main(
    concepts_path="data/table/materials-science.llama.works.csv",
    lookup_path="data/table/lookup/lookup_large.csv",
    output_path="data/embeddings/large/",
    log_to_stdout=False,
    step_size=500,
    start=0,
    end=80_000,
):
    global logger, get_embeddings, get_token_ids
    logger = setup_logger(logging.INFO, log_to_stdout=log_to_stdout)
    logger.info("Prepare dataframe")

    df = prepare_dataframe(
        df=pd.read_csv(concepts_path),
        lookup_df=pd.read_csv(lookup_path),
        cols=["id", "abstract", "concepts"],
    )

    logger.info("Setup model")
    tokenizer, model = setup_model("m3rg-iitd/matscibert")
    get_embeddings = init_get_embeddings(model, tokenizer)
    get_token_ids = init_get_token_ids(tokenizer)

    logger.info("Generate word embeddings")

    for i in range(start, end, step_size):
        logger.info(f"Process {i} to {i+step_size}...")
        partial_df = df[i : i + step_size]
        store = process_works(partial_df, desc=f"Generate embeddings ({i})")
        logger.info("Save embeddings")
        save_path = os.path.join(output_path, f"embeddings_{i:06d}.pkl.gz")
        save_compressed(store, save_path)


if __name__ == "__main__":
    fire.Fire(main)
