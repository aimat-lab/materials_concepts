import pandas as pd
from utils import prepare_dataframe
import os
import pickle, gzip
from tqdm import tqdm
import logging
import sys
import torch
import fire

DIM_EMBEDDING = 768


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


class EmbeddingAverager:
    def __init__(self):
        self.storage = {}

    def __len__(self):
        return len(self.storage)

    def __setitem__(self, key, value):
        if key not in self.storage:
            self.storage[key] = (value, 1)
        else:
            old_value, count = self.storage[key]
            self.storage[key] = (old_value + value, count + 1)

    def __getitem__(self, key):
        value, count = self.storage[key]
        return value / count

    def __contains__(self, key):
        return key in self.storage

    def keys(self):
        return self.storage.keys()

    def save(self, path, concept_mapping=None):
        translated = self.storage

        if concept_mapping:
            translated = {
                concept_mapping[concept]: self.__getitem__(concept)
                for concept in self.storage.keys()
            }

        with gzip.open(path, "wb") as f:
            pickle.dump(translated, f)

    def null_ratio(self):
        return round(
            sum(
                [
                    1
                    for embs, _ in self.storage.values()
                    if all(embs == torch.zeros(DIM_EMBEDDING))
                ]
            )
            / len(self.stores),
            3,
        )


class DataReader:
    def __init__(self, embeddings_path, df, logger):
        self.embeddings_path = embeddings_path
        self.df = df
        self.lookup = {id: concepts for id, concepts in zip(df["id"], df["concepts"])}
        self.files = sorted(
            [f for f in os.listdir(embeddings_path) if f.endswith(".pkl.gz")],
            key=lambda path: int(path[:-7].rsplit("_")[1]),
        )
        self.logger = logger

    def get_averaged_concept_embeddings(self, concepts_filter, until_year=None):
        averaged_embeddings = EmbeddingAverager()

        # create date from year
        if until_year:
            until_date = pd.to_datetime(f"{until_year+1}-01-01")
            ids = set(self.df[self.df.publication_date < until_date]["id"])
        else:
            ids = set(self.df["id"])

        self.logger.info(
            f"Averaging concepts of: {len(ids)} works (total: {len(self.df)})"
        )

        # read file by file
        for file in self.files:
            self.logger.info(f"Reading file: {file}")
            chunk = DataReader.load_compressed(os.path.join(self.embeddings_path, file))

            # iterate over works in file
            for id, embeddings in tqdm(chunk.items(), desc=f"Processing works"):
                self.logger.debug(f"Processing work - id: {id}")
                if id not in ids:
                    continue

                # filter embeddings according to concepts)
                assert len(embeddings) == len(self.lookup[id])

                for emb, con in zip(embeddings, self.lookup[id]):
                    if con in concepts_filter:
                        averaged_embeddings[con] = emb

        return averaged_embeddings

    @staticmethod
    def load_compressed(path):
        with open(path, "rb") as f:
            compressed = f.read()
        return pickle.loads(gzip.decompress(compressed))


def main(
    concepts_path="data/table/materials-science.llama.works.csv",
    lookup_path="data/table/lookup/lookup_large.csv",
    filter_path="data/table/lookup/lookup_small.csv",
    embeddings_dir="data/embeddings/large/",
    output_path="data/model/con_embs/",
    until_year=2016,
):
    logger = setup_logger(level=logging.INFO, log_to_stdout=True)

    df = prepare_dataframe(
        df=pd.read_csv(concepts_path),
        lookup_df=pd.read_csv(lookup_path),
        cols=["id", "concepts", "publication_date"],
    )

    filter_df = pd.read_csv(filter_path)

    concept_filter = set(filter_df["concept"])

    dr = DataReader(embeddings_dir, df, logger)
    averaged_embeddings = dr.get_averaged_concept_embeddings(concept_filter, until_year)

    concept_to_id = {c: i for i, c in zip(filter_df.id, filter_df.concept)}

    output_path = os.path.join(output_path, f"av_embs_{until_year}.pkl.gz")
    averaged_embeddings.save(
        output_path,
        concept_mapping=concept_to_id,
    )
    logger.info(f"Saved {len(averaged_embeddings)} embeddings to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
