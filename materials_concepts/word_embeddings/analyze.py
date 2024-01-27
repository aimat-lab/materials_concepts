import pandas as pd
from utils import prepare_dataframe
import os
import pickle, gzip
from tqdm import tqdm
import logging
import sys
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

    def get_averaged_concept_embeddings(self):
        store = {}
        # read file by file
        for file in self.files:
            self.logger.info(f"Reading file: {file}")
            chunk = DataReader.load_compressed(os.path.join(self.embeddings_path, file))

            # iterate over works in file
            for id, embeddings in tqdm(chunk.items(), desc=f"Processing works"):
                self.logger.debug(f"Processing work - id: {id}")

                # filter embeddings according to concepts)

                store[id] = {
                    "len_embeddings": len(embeddings),
                    "len_concepts": len(self.lookup[id]),
                }

        return store

    @staticmethod
    def load_compressed(path):
        with open(path, "rb") as f:
            compressed = f.read()
        return pickle.loads(gzip.decompress(compressed))


def main(
    concepts_path="data/table/materials-science.llama.works.csv",
    lookup_path="data/table/lookup/lookup_large.csv",
    embeddings_dir="data/embeddings/large/",
):
    logger = setup_logger(level=logging.INFO, log_to_stdout=True)

    df = prepare_dataframe(
        df=pd.read_csv(concepts_path),
        lookup_df=pd.read_csv(lookup_path),
        cols=["id", "concepts", "publication_date"],
    )

    dr = DataReader(embeddings_dir, df, logger)
    store = dr.get_averaged_concept_embeddings()
    analysis = pd.DataFrame(store).T
    logger.info((analysis.len_embeddings != analysis.len_concepts).value_counts())


if __name__ == "__main__":
    fire.Fire(main)
