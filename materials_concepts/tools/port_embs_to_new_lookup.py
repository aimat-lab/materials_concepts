import pandas as pd
from ast import literal_eval
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


def prepare_dataframe(df, lookup_df, cols):
    lookup = {key: True for key in lookup_df["concept"]}

    df.abstract = df.abstract.str.lower()
    df.llama_concepts = df.llama_concepts.apply(literal_eval).apply(
        lambda x: list({c.lower() for c in x if lookup.get(c.lower())})
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

    def port_embeddings(
        self,
        to_keep,
        output_path,
    ):  # each pkl is a dict of {work_id: tensor[x, 768]}
        self.logger.info(f"Porting embeddings...")

        # read file by file
        for file in self.files:
            self.logger.info(f"Reading file: {file}")
            chunk = DataReader.load_compressed(os.path.join(self.embeddings_path, file))

            store = {}

            # iterate over works in file
            for id, embeddings in tqdm(chunk.items(), desc=f"Processing works"):
                self.logger.debug(f"Processing work - id: {id}")

                # filter embeddings according to concepts)
                assert len(embeddings) == len(self.lookup[id])

                store[id] = torch.stack(
                    [
                        emb
                        for emb, con in zip(embeddings, self.lookup[id])
                        if con in to_keep
                    ]
                )
                self.logger.debug(
                    f"Kept {len(store[id])} embeddings of {len(embeddings)}"
                )

            new_file = os.path.join(output_path, file)
            self.logger.info(f"Saving to {new_file}")
            self.save_compressed(store, new_file)

    @staticmethod
    def save_compressed(obj, path):
        with gzip.open(path, "wb") as f:
            f.write(pickle.dumps(obj))

    @staticmethod
    def load_compressed(path):
        with open(path, "rb") as f:
            compressed = f.read()
        return pickle.loads(gzip.decompress(compressed))


def main(
    concepts_path="data/table/materials-science.llama.works.csv",
    lookup_path="data/table/lookup/lookup_large_legacy.csv",
    keep_path="data/table/lookup/lookup_large.csv",
    embeddings_dir="data/embeddings/large_legacy/",
    output_path="data/embeddings/large/",
):
    logger = setup_logger(level=logging.INFO, log_to_stdout=True)

    df = prepare_dataframe(
        df=pd.read_csv(concepts_path),
        lookup_df=pd.read_csv(lookup_path),
        cols=["id", "concepts", "publication_date"],
    )

    dr = DataReader(embeddings_dir, df, logger)
    dr.port_embeddings(set(pd.read_csv(keep_path)["concept"]), output_path)


if __name__ == "__main__":
    fire.Fire(main)
