import pandas as pd
from ast import literal_eval
import os
import pickle, gzip
from tqdm import tqdm
import logging
import sys


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


def prepare_dataframe(df, lookup_df, cols):
    lookup = {key: True for key in lookup_df["concept"]}

    df.abstract = df.abstract.str.lower()
    df.llama_concepts = df.llama_concepts.apply(literal_eval).apply(
        lambda x: list({c.lower() for c in x if lookup.get(c)})
    )

    df.elements = df.elements.apply(
        lambda str: list({e.lower() for e in str.split(",") if lookup.get(e)})
        if not pd.isna(str)
        else []
    )

    df.publication_date = pd.to_datetime(df.publication_date)

    df.concepts = df.llama_concepts + df.elements
    df.concepts = df.concepts.apply(lambda x: sorted(x))  # sort

    return df[cols]


class EmbeddingAverager:
    def __init__(self):
        self.store = {}

    def __setitem__(self, key, value):
        if key not in self.store:
            self.store[key] = (value, 1)
        else:
            old_value, count = self.store[key]
            self.store[key] = (old_value + value, count + 1)

    def __getitem__(self, key):
        value, count = self.store[key]
        return value / count

    def __contains__(self, key):
        return key in self.store


class DataReader:
    def __init__(self, embeddings_path, df, logger):
        self.embeddings_path = embeddings_path
        self.df = df
        self.files = [f for f in os.listdir(embeddings_path) if f.endswith(".pkl.gz")]
        self.logger = logger

    def get_averaged_concept_embeddings(self, concepts_filter, until_year=None):
        averaged_embeddings = EmbeddingAverager()

        # create date from year
        if until_year:
            until_date = pd.to_datetime(until_year, format="%Y")
            ids = set(self.df[self.df.publication_date <= until_date]["id"])
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

                print(self.df[self.df.id == id])
                # filter embeddings according to concepts)
                assert len(embeddings) == len(
                    self.df[self.df.id == id].loc[150][
                        "concepts"
                    ]  # TODO: write wrapper for nice get access
                )

                for emb, con in zip(
                    embeddings, self.df[self.df.id == id].loc[150]["concepts"]
                ):
                    if con in concepts_filter:
                        averaged_embeddings[con] = emb

        return averaged_embeddings

    @staticmethod
    def load_compressed(path):
        with open(path, "rb") as f:
            compressed = f.read()
        return pickle.loads(gzip.decompress(compressed))


concepts_path = "data/table/materials-science.llama.works.csv"
lookup_path = "data/table/lookup/lookup_large.csv"
logger = setup_logger(level=logging.DEBUG, log_to_stdout=True)

df = prepare_dataframe(
    df=pd.read_csv(concepts_path),
    lookup_df=pd.read_csv(lookup_path),
    cols=["id", "concepts", "publication_date"],
)

concept_filter = set(pd.read_csv(lookup_path)["concept"])

dr = DataReader("data/embeddings/large/", df, logger)
averaged_embeddings = dr.get_averaged_concept_embeddings(concept_filter, 2016)
