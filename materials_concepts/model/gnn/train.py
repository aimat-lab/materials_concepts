import gzip
import logging
import os
import pickle
import random
import sys
from dataclasses import dataclass
from importlib import reload
from typing import Any

import fire
import numpy as np
import torch
from torch import nn

from materials_concepts.model.graph import Graph
from materials_concepts.model.metrics import test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_logger(file, level=logging.INFO, log_to_stdout=True):
    logger = logging.getLogger()
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


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_compressed(path: str | None):
    if not path:
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def flatten(t):
    return [item for sublist in t for item in sublist]


def sample_pair_batch(y: torch.Tensor, batch_size: int, pos_ratio: float = 0.5):
    pos_indices = torch.where(y == 1)[0]
    neg_indices = torch.where(y == 0)[0]

    amt_pos = int(batch_size * pos_ratio)
    amt_neg = batch_size - amt_pos

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        # fall back to uniform sampling
        return torch.randint(0, len(y), (batch_size,))

    i_pos = torch.randint(0, len(pos_indices), (amt_pos,))
    i_neg = torch.randint(0, len(neg_indices), (amt_neg,))

    batch_indices = torch.cat([pos_indices[i_pos], neg_indices[i_neg]])
    batch_indices = batch_indices[torch.randperm(len(batch_indices))]
    return batch_indices


@dataclass
class BatchSample:
    seeds: np.ndarray  # unique seed node ids (u and v endpoints)
    seeds_u: np.ndarray  # per-pair u endpoints in same id space as seeds
    seeds_v: np.ndarray  # per-pair v endpoints
    neighbors1: dict[int, np.ndarray]  # for seed nodes only
    neighbors2: dict[int, np.ndarray]  # for 1-hop sampled neighbor nodes only
    nodes0: np.ndarray  # seed nodes
    nodes1: np.ndarray  # 1-hop sampled neighbor nodes
    nodes2: np.ndarray  # 2-hop sampled neighbor nodes


class CSRNeighborSampler:
    def __init__(self, adj_csr, rng: np.random.Generator):
        # scipy.sparse.csr_matrix
        self.indptr = adj_csr.indptr
        self.indices = adj_csr.indices
        self.rng = rng

    def sample_neighbors(self, nodes: np.ndarray, fanout: int) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        if fanout <= 0:
            for n in nodes:
                out[int(n)] = np.empty((0,), dtype=np.int64)
            return out

        for n in nodes:
            n_int = int(n)
            start = self.indptr[n_int]
            end = self.indptr[n_int + 1]
            neigh = self.indices[start:end]
            if neigh.size == 0:
                out[n_int] = np.empty((0,), dtype=np.int64)
                continue
            if neigh.size <= fanout:
                out[n_int] = neigh.astype(np.int64, copy=False)
            else:
                # without replacement
                choice = self.rng.choice(neigh.size, size=fanout, replace=False)
                out[n_int] = neigh[choice].astype(np.int64, copy=False)
        return out


def _parse_config_str(value: str | None, defaults: dict[str, Any]) -> dict[str, Any]:
    """Parse a simple comma-separated key=value string into a dict.

    Example: "batch_size=512,lr=1e-3,pos_ratio=0.3"
    """
    out = dict(defaults)
    if not value:
        return out

    items = [x.strip() for x in value.split(",") if x.strip()]
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid config item '{item}'. Expected key=value.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k not in out:
            raise ValueError(f"Unknown config key '{k}'. Allowed: {sorted(out.keys())}")

        # basic type parsing based on default type
        default_val = out[k]
        if isinstance(default_val, bool):
            out[k] = v.lower() in {"1", "true", "yes", "y", "on"}
        elif isinstance(default_val, int) and not isinstance(default_val, bool):
            out[k] = int(float(v))
        elif isinstance(default_val, float):
            out[k] = float(v)
        else:
            out[k] = v

    return out


def build_pair_batch_sample(
    pairs: np.ndarray,
    fanout1: int,
    fanout2: int,
    sampler: CSRNeighborSampler,
):
    # pairs is shape (B, 2) with global node ids
    u = pairs[:, 0].astype(np.int64, copy=False)
    v = pairs[:, 1].astype(np.int64, copy=False)
    nodes0 = np.unique(np.concatenate([u, v]))

    neighbors1 = sampler.sample_neighbors(nodes0, fanout1)
    nodes1 = (
        np.unique(np.concatenate([n for n in neighbors1.values() if n.size > 0]))
        if len(neighbors1)
        else np.empty((0,), dtype=np.int64)
    )

    neighbors2 = sampler.sample_neighbors(nodes1, fanout2) if nodes1.size else {}
    nodes2 = (
        np.unique(np.concatenate([n for n in neighbors2.values() if n.size > 0]))
        if len(neighbors2)
        else np.empty((0,), dtype=np.int64)
    )

    return BatchSample(
        seeds=nodes0,
        seeds_u=u,
        seeds_v=v,
        neighbors1=neighbors1,
        neighbors2=neighbors2,
        nodes0=nodes0,
        nodes1=nodes1,
        nodes2=nodes2,
    )


class GraphSAGE2Layer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def _mean_agg(self, h: torch.Tensor, neigh_indices: list[list[int]]):
        # h: (N, D), neigh_indices: list of lists of indices into h
        # returns (len(neigh_indices), D)
        out = []
        d = h.shape[1]
        zeros = torch.zeros((d,), device=h.device, dtype=h.dtype)
        for idxs in neigh_indices:
            if not idxs:
                out.append(zeros)
            else:
                out.append(h[idxs].mean(dim=0))
        return torch.stack(out, dim=0)

    def forward(
        self,
        x_all: torch.Tensor,
        bottom_nodes: np.ndarray,
        middle_nodes: np.ndarray,
        seed_nodes: np.ndarray,
        neighbors_for_layer1: dict[int, np.ndarray],
        neighbors_for_layer1_alt: dict[int, np.ndarray],
        neighbors_for_layer2: dict[int, np.ndarray],
    ):
        """Compute embeddings for seed_nodes with 2-layer sampling.

        x_all: node features for bottom_nodes, already gathered (len(bottom_nodes), in_dim)
        bottom_nodes: global ids for x_all rows
        middle_nodes: global ids that get layer1 embeddings (seeds ∪ nodes1)
        seed_nodes: global ids to return final embeddings for
        neighbors_for_layer1: mapping for seeds -> sampled 1-hop neighbors (subset of nodes1)
        neighbors_for_layer1_alt: mapping for nodes1 -> sampled neighbors for layer1 (subset nodes2)
        neighbors_for_layer2: mapping for seeds -> sampled neighbors for layer2 (same as neighbors_for_layer1)
        """
        bottom_index = {int(n): i for i, n in enumerate(bottom_nodes)}

        # layer1 embeddings for middle_nodes
        self_idx = [bottom_index[int(n)] for n in middle_nodes]
        neigh_lists = []
        for n in middle_nodes:
            n_int = int(n)
            if n_int in neighbors_for_layer1:
                neigh = neighbors_for_layer1[n_int]
            else:
                neigh = neighbors_for_layer1_alt.get(
                    n_int, np.empty((0,), dtype=np.int64)
                )
            neigh_lists.append(
                [bottom_index[int(v)] for v in neigh if int(v) in bottom_index]
            )

        h_self = x_all[self_idx]
        h_neigh = self._mean_agg(x_all, neigh_lists)
        h1 = self.act(self.lin1(torch.cat([h_self, h_neigh], dim=1)))
        h1 = self.dropout(h1)

        middle_index = {int(n): i for i, n in enumerate(middle_nodes)}

        # layer2 embeddings for seed_nodes
        seed_self_idx = [middle_index[int(n)] for n in seed_nodes]
        seed_neigh_lists = []
        for n in seed_nodes:
            neigh = neighbors_for_layer2.get(int(n), np.empty((0,), dtype=np.int64))
            seed_neigh_lists.append(
                [middle_index[int(v)] for v in neigh if int(v) in middle_index]
            )

        h2_self = h1[seed_self_idx]
        h2_neigh = self._mean_agg(h1, seed_neigh_lists)
        z = self.act(self.lin2(torch.cat([h2_self, h2_neigh], dim=1)))
        return z


def build_binary_adj_for_year(graph: Graph, year: int):
    # full=True equivalent: include gaps up to max node id
    edges = graph.get_until_year(year)
    dim = int(np.max(graph.vertices)) + 1
    return Graph.build_adj_matrix(edges, binary=True, dim=dim)


def eval_pairs(
    model: GraphSAGE2Layer,
    sampler: CSRNeighborSampler,
    v_features: np.ndarray,
    pairs: np.ndarray,
    labels: np.ndarray,
    fanout1: int,
    fanout2: int,
    batch_size: int,
):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            sample = build_pair_batch_sample(batch_pairs, fanout1, fanout2, sampler)

            bottom_nodes = np.unique(
                np.concatenate([sample.nodes0, sample.nodes1, sample.nodes2])
            )
            middle_nodes = np.unique(np.concatenate([sample.nodes0, sample.nodes1]))

            x_all = torch.tensor(
                v_features[bottom_nodes], dtype=torch.float32, device=device
            )

            z_seeds = model(
                x_all=x_all,
                bottom_nodes=bottom_nodes,
                middle_nodes=middle_nodes,
                seed_nodes=sample.nodes0,
                neighbors_for_layer1=sample.neighbors1,
                neighbors_for_layer1_alt=sample.neighbors2,
                neighbors_for_layer2=sample.neighbors1,
            )

            seed_index = {int(n): j for j, n in enumerate(sample.nodes0)}
            u_idx = torch.tensor(
                [seed_index[int(u)] for u in sample.seeds_u], device=device
            )
            v_idx = torch.tensor(
                [seed_index[int(v)] for v in sample.seeds_v], device=device
            )
            z_u = z_seeds[u_idx]
            z_v = z_seeds[v_idx]
            logits = (z_u * z_v).sum(dim=1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_scores.extend(probs.tolist())

    predictions = np.array(all_scores)
    auc, _, confusion_matrix = test(
        torch.tensor(labels, dtype=torch.float32), predictions, threshold=0.5
    )
    return auc, confusion_matrix


def main(
    graph_path="data/graph/edges.pkl",
    data_path="data/model/data.pkl",
    v_features_path="data/model/combi/matrices_2016.pkl.gz",
    year_start_train=2016,
    train=None,
    model=None,
    sampling=None,
    seed=42,
    log_file="logs/gnn_train.log",
    save_model_path=None,
):
    """Train a 2-layer GraphSAGE-style GNN for temporal link prediction.

    Supervision mirrors the existing pipeline: pairs (u,v) with labels indicating whether
    the edge appears in the future window, while message passing uses ONLY the past graph
    at year_start_train.
    """
    reload(logging)
    global logger
    logger = setup_logger(file=log_file, level=logging.INFO, log_to_stdout=True)

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    rng = np.random.default_rng(seed)

    train_cfg = _parse_config_str(
        train,
        defaults={
            "batch_size": 512,
            "pos_ratio": 0.3,
            "num_epochs": 50,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "log_interval": 5,
            "eval_batch_size": 4096,
        },
    )
    model_cfg = _parse_config_str(
        model,
        defaults={
            "hidden_dim": 64,
            "out_dim": 64,
            "dropout": 0.1,
        },
    )
    sampling_cfg = _parse_config_str(
        sampling,
        defaults={
            "fanout1": 15,
            "fanout2": 10,
        },
    )

    logger.info(f"device: {device}")
    logger.info(f"seed: {seed}")
    logger.info(f"year_start_train: {year_start_train}")
    logger.info(f"fanout: ({sampling_cfg['fanout1']}, {sampling_cfg['fanout2']})")

    logger.info("Loading pair dataset")
    data = load_pickle(data_path)

    logger.info("Loading node features (v_features)")
    feats = load_compressed(v_features_path)
    if not feats or "v_features" not in feats:
        raise ValueError(f"Expected v_features in compressed file: {v_features_path}")
    v_features = feats["v_features"]

    logger.info("Building past-graph adjacency (binary CSR)")
    graph = Graph(graph_path)
    adj = build_binary_adj_for_year(graph, year_start_train)
    sampler = CSRNeighborSampler(adj, rng=rng)

    x_train = np.asarray(data["X_train"], dtype=np.int64)
    y_train = np.asarray(data["y_train"], dtype=np.float32)
    x_val = np.asarray(data.get("X_val", data.get("X_test")), dtype=np.int64)
    y_val = np.asarray(data.get("y_val", data.get("y_test")), dtype=np.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    in_dim = int(v_features.shape[1])
    model = GraphSAGE2Layer(
        in_dim=in_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()

    logger.info(
        f"Model: GraphSAGE2Layer(in_dim={in_dim}, hidden_dim={model_cfg['hidden_dim']}, out_dim={model_cfg['out_dim']})"
    )
    logger.info(f"Train pairs: {len(x_train)} | Val pairs: {len(x_val)}")

    for epoch in range(1, int(train_cfg["num_epochs"]) + 1):
        model.train()

        batch_idx = sample_pair_batch(
            y_train_t,
            batch_size=int(train_cfg["batch_size"]),
            pos_ratio=float(train_cfg["pos_ratio"]),
        )
        batch_pairs = x_train[batch_idx.numpy()]
        batch_labels = y_train_t[batch_idx].to(device)

        sample = build_pair_batch_sample(
            batch_pairs,
            int(sampling_cfg["fanout1"]),
            int(sampling_cfg["fanout2"]),
            sampler,
        )

        bottom_nodes = np.unique(
            np.concatenate([sample.nodes0, sample.nodes1, sample.nodes2])
        )
        middle_nodes = np.unique(np.concatenate([sample.nodes0, sample.nodes1]))

        x_all = torch.tensor(
            v_features[bottom_nodes], dtype=torch.float32, device=device
        )
        z_seeds = model(
            x_all=x_all,
            bottom_nodes=bottom_nodes,
            middle_nodes=middle_nodes,
            seed_nodes=sample.nodes0,
            neighbors_for_layer1=sample.neighbors1,
            neighbors_for_layer1_alt=sample.neighbors2,
            neighbors_for_layer2=sample.neighbors1,
        )

        seed_index = {int(n): j for j, n in enumerate(sample.nodes0)}
        u_idx = torch.tensor(
            [seed_index[int(u)] for u in sample.seeds_u], device=device
        )
        v_idx = torch.tensor(
            [seed_index[int(v)] for v in sample.seeds_v], device=device
        )

        z_u = z_seeds[u_idx]
        z_v = z_seeds[v_idx]
        logits = (z_u * z_v).sum(dim=1)

        loss = criterion(logits, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % int(train_cfg["log_interval"]) == 0:
            auc, (tn, fp, fn, tp) = eval_pairs(
                model=model,
                sampler=sampler,
                v_features=v_features,
                pairs=x_val,
                labels=y_val,
                fanout1=int(sampling_cfg["fanout1"]),
                fanout2=int(sampling_cfg["fanout2"]),
                batch_size=int(train_cfg["eval_batch_size"]),
            )
            logger.info(
                f"Epoch: {epoch}, Loss: {loss.item():.4f}, AUC: {auc:.4f}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}"
            )

    if save_model_path:
        logger.info(f"Saving model to {save_model_path}")
        os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_model_path)


if __name__ == "__main__":
    fire.Fire(main)
