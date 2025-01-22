import os
import pickle
import gzip
from tqdm import tqdm

path = "data/embeddings/final_fix/"


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


for i in tqdm(range(0, 222000, 500)):
    if not os.path.exists(os.path.join(path, f"embeddings_{i:06d}.pkl.gz")):
        print(i)
