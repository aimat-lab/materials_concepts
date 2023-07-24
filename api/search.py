from transformers import AutoTokenizer, AutoModel
import pickle
import gzip
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

CONCEPT_PATH = "data/embeddings/embeddings_full.pkl.gz"


def setup_model(model_name):
    print("Setting up model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return tokenizer, model


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


def convert_tensors_to_arrays(data):
    return {k: v.numpy() for k, v in data.items()}


class Search:
    def __init__(self, data, k, model_name):
        self.data = data
        self.k = k + 1  # +1 because the first result is the query itself
        self.values = np.array(list(self.data.values()))
        self.keys = np.array(list(self.data.keys()))
        print("Fitting knn")
        self.tokenizer, self.model = setup_model(model_name)

    def _nn_search(self, string, k=5):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(self.values)
        if string not in self.keys:
            emb = self._get_embeddings(string)
        else:
            emb = self.data[string]

        return self._get_knn(emb, nbrs)

    def _plain_search(self, string):
        return [k for k in self.keys if string in k]

    def _get_embeddings(self, string):
        tokens = self.tokenizer(string)["input_ids"]

        output = self.model(torch.tensor([tokens]))

        embeddings = output.last_hidden_state.squeeze()
        embeddings = embeddings[1:-1]  # remove [CLS] and [SEP]
        phrase_embedding = torch.mean(embeddings, dim=0).detach().numpy()

        return phrase_embedding

    def _get_knn(self, emb, nbrs):
        distances, indices = nbrs.kneighbors([emb])
        return [(self.keys[i], round(d, 3)) for i, d in zip(indices[0], distances[0])]

    def search(self, string, semantic_search=False):
        if semantic_search:
            return self._nn_search(string)
        else:
            return self._plain_search(string)


print("Loading data")
data = convert_tensors_to_arrays(load_compressed(CONCEPT_PATH))

search = Search(
    data,
    k=6,
    model_name="m3rg-iitd/matscibert",
)
