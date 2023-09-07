import pickle
import gzip
import torch
from sklearn.decomposition import PCA
from fire import Fire


def load_compressed(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_compressed(path, data):
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)


def main(input_file="", output_file=""):
    data = load_compressed(input_file)

    matrix = torch.stack(tuple(data.values()))
    pca = PCA(n_components=5)
    reduced_embeddings = pca.fit_transform(matrix.numpy())

    # restore dict like structure
    reduced_data = {k: v for k, v in zip(data.keys(), reduced_embeddings)}

    save_compressed(output_file, reduced_data)


if __name__ == "__main__":
    Fire(main)
