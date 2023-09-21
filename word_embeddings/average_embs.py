import pandas as pd
from utils import prepare_dataframe
import os
import pickle, gzip
from tqdm import tqdm
import logging
import sys
import fire
import torch

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
    def __init__(self, only_average_contained=False):
        self.only_average_contained = only_average_contained
        self.storage = {}

    def __len__(self):
        return len(self.storage)

    def __setitem__(self, key, value):
        if key not in self.storage:
            self.storage[key] = (value, 1)
        else:
            old_value, count = self.storage[key]
            self.storage[key] = (old_value + value, count + 1)

    def add(self, key, value, contained):
        """If a concept is contained, we store these embeddings separately, as they are exact.
        If the concept is not contained, the embedding is only averaged, which isn't that valuable.
        """
        if key not in self.storage:
            self.storage[key] = {
                "exact_matches": (torch.zeros(DIM_EMBEDDING), 0),
                "averaged": (torch.zeros(DIM_EMBEDDING), 0),
            }

        to_store = "exact_matches" if contained else "averaged"

        old_value, count = self.storage[key][to_store]

        self.storage[key][to_store] = (old_value + value, count + 1)

    def __getitem__(self, key):
        """Returns the averaged embedding for a concept. If there is at least one exact match,
        the average of these exact matches is returned.
        Otherwise, the already averaged embeddings are averaged again.
        """
        if self.only_average_contained:
            _, exact_count = self.storage[key]["exact_matches"]

            to_retrieve = "exact_matches" if exact_count > 0 else "averaged"

            value, count = self.storage[key][to_retrieve]
            return value / count
        else:
            store = self.storage[key]
            exact_value, exact_count = store["exact_matches"]
            averaged_value, averaged_count = store["averaged"]

            total_count = exact_count + averaged_count

            return (exact_value + averaged_value) / total_count

    def __contains__(self, key):
        return key in self.storage

    def keys(self):
        return self.storage.keys()

    def save(self, path, concept_mapping=None):
        if concept_mapping:
            translated = {
                concept_mapping[concept]: self.__getitem__(concept)
                for concept in self.storage.keys()
            }
        else:
            translated = {
                key: self.__getitem__(key) for key in self.storage.keys()
            }  # this might seem unnecessary but we use this getitem access to convert the (tensor, count) tuple to a single tensor

        with gzip.open(path, "wb") as f:
            pickle.dump(translated, f)


class DataReader:
    def __init__(self, embeddings_path, df, logger):
        self.embeddings_path = embeddings_path
        self.df = df
        self.files = sorted(
            [f for f in os.listdir(embeddings_path) if f.endswith(".pkl.gz")],
            key=lambda path: int(path[:-7].rsplit("_")[1]),
        )
        self.logger = logger

    def get_averaged_concept_embeddings(
        self, concepts_filter, until_year=None, only_average_contained=False
    ):
        averaged_embeddings = EmbeddingAverager(only_average_contained)

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

                # select field "concept_contained" where df.id == id
                containment = self.df[self.df.id == id]["concept_contained"].iloc[0]
                self.logger.debug(f"Containment: {containment}")

                for con, emb in embeddings.items():
                    if con in concepts_filter:
                        averaged_embeddings.add(con, emb, containment[con])

        return averaged_embeddings

    @staticmethod
    def load_compressed(path):
        with open(path, "rb") as f:
            compressed = f.read()
        return pickle.loads(gzip.decompress(compressed))


def compute_containment(df: pd.DataFrame):
    df["concept_contained"] = df.apply(
        lambda row: {
            concept: concept.lower() in row["abstract"].lower()
            for concept in row["concepts"]
        },
        axis=1,
    )


def main(
    concepts_path="data/table/materials-science.llama.works.csv",
    lookup_path="data/table/lookup/lookup_large.csv",
    filter_path="data/table/lookup/lookup_small.csv",
    embeddings_dir="data/embeddings/large/",
    output_path="data/model/con_embs/av_embs_small_2016.pkl.gz",
    store_concepts_ids=False,
    until_year=2016,
    only_average_contained=False,
):
    logger = setup_logger(level=logging.INFO, log_to_stdout=True)

    df = prepare_dataframe(
        df=pd.read_csv(concepts_path),
        lookup_df=pd.read_csv(lookup_path),
        cols=["id", "concepts", "publication_date", "abstract"],
    )

    compute_containment(df)

    filter_df = pd.read_csv(filter_path)

    concept_filter = set(filter_df["concept"])

    dr = DataReader(embeddings_dir, df, logger)
    averaged_embeddings = dr.get_averaged_concept_embeddings(
        concept_filter, until_year, only_average_contained=only_average_contained
    )

    concept_mapping = None

    if store_concepts_ids:
        concept_mapping = {c: i for i, c in zip(filter_df.id, filter_df.concept)}

    averaged_embeddings.save(
        output_path,
        concept_mapping=concept_mapping,
    )
    logger.info(f"Saved {len(averaged_embeddings)} embeddings to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
