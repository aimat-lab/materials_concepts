import pickle
import gzip
import torch
from sklearn.decomposition import PCA
from fire import Fire
from loguru import logger
import umap


def load_compressed(path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_compressed(path, data) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)


def main(input_file="", output_file="", n_components=2):
    logger.info(f"Loading {input_file}")
    data = load_compressed(input_file)

    matrix = torch.stack(tuple(data.values()))

    logger.info(f"Reducing dimensionality to {n_components}")
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(matrix.numpy())

    # restore dict like structure
    reduced_data = {k: v for k, v in zip(data.keys(), reduced_embeddings)}

    logger.info(f"Saving to {output_file}")
    save_compressed(output_file, reduced_data)


def main_umap(input_file="", output_file="", n_components=2):
    logger.info(f"Loading {input_file}")
    data = load_compressed(input_file)

    matrix = torch.stack(tuple(data.values()))

    logger.info(f"Reducing dimensionality to {n_components} using UMAP")
    umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = umap_reducer.fit_transform(matrix.numpy())

    # restore dict-like structure
    reduced_data = {k: v for k, v in zip(data.keys(), reduced_embeddings)}

    logger.info(f"Saving to {output_file}")
    save_compressed(output_file, reduced_data)


if __name__ == "__main__":
    Fire(main_umap)
