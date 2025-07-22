import pandas as pd
import pickle
import gzip


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


def load_lookup(lookup_path):
    return pd.read_csv(lookup_path)
