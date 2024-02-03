import gzip
import logging
import pickle
import sys
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def apply_in_parallel(df, func, n_jobs):
    tasks = np.array_split(df, n_jobs, axis=0)  # split df along row axis
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        result = executor.map(func, tasks)

    return pd.concat(result)


def load_pickle(path: Path | str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path | str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_compressed(path: Path | str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_compressed(obj, path: Path | str):
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


class Timer:
    start: datetime

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        print(
            f"{self.name} time elapsed: {(datetime.now() - self.start).seconds} seconds..."
        )


def remove_non_ascii(text):
    """
    Removes non-ASCII characters from a string.
    """
    return "".join([char for char in text if ord(char) < 128])


def make_valid_filename(filename):
    """
    Removes invalid characters from a string to create a valid filename.
    """

    filename = remove_non_ascii(filename)

    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "")

    # Remove leading/trailing whitespace and dots
    filename = filename.strip(". ")

    # Truncate the filename to 255 characters (max filename length on most file systems)
    filename = filename[:255]

    filename = filename.replace(" ", "-")

    return filename


def setup_logger(logger: logging.Logger, file, level=logging.INFO, log_to_stdout=True):
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    if log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def flatten(t):
    return [item for sublist in t for item in sublist]


def prepare_dataframe(df, lookup_df, cols):
    lookup = {key: True for key in lookup_df["concept"]}

    df.llama_concepts = df.llama_concepts.apply(literal_eval).apply(
        lambda x: list({c.lower() for c in x if lookup.get(c.lower())})
    )

    df.elements = df.elements.apply(
        lambda str: list({e for e in str.split(",") if lookup.get(e)})
        if not pd.isna(str)
        else []
    )

    df.publication_date = pd.to_datetime(df.publication_date)

    df.concepts = df.llama_concepts + df.elements
    df.concepts = df.concepts.apply(lambda x: sorted(x))  # sort

    return df[cols]
