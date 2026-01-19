import gzip
import logging
import os
import pickle
import random
import sys
from importlib import reload
from typing import Any

import fire
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from materials_concepts.model.graph import Graph
from materials_concepts.model.metrics import test


def _maybe_init_wandb(
    wandb_cfg: dict[str, Any],
    config: dict[str, Any],
    logger: logging.Logger,
):
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "W&B logging requested but 'wandb' is not installed.\n"
            "Install with: python -m pip install wandb\n"
            "Or disable with: --wandb=enabled=false"
        ) from e

    tags_raw = str(wandb_cfg.get("tags", "")).strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else None
    mode = str(wandb_cfg.get("mode", "online")).strip() or "online"
    fail_fast = bool(wandb_cfg.get("fail_fast", False))

    init_kwargs: dict[str, Any] = {
        "project": str(wandb_cfg.get("project", "materials_concepts")),
        "entity": str(wandb_cfg.get("entity", "")) or None,
        "name": str(wandb_cfg.get("name", "")) or None,
        "group": str(wandb_cfg.get("group", "")) or None,
        "job_type": str(wandb_cfg.get("job_type", "train")) or None,
        "tags": tags,
        "mode": mode,
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:
        # Common on clusters: no internet / blocked outbound, or permission issues.
        # Keep training running unless explicitly asked to fail-fast.
        logger.error(
            "W&B init failed (project=%s, entity=%s, mode=%s): %s: %s",
            init_kwargs.get("project"),
            init_kwargs.get("entity"),
            init_kwargs.get("mode"),
            type(e).__name__,
            e,
        )
        if fail_fast:
            raise

        if str(init_kwargs.get("mode", "")) != "offline":
            try:
                logger.info("Retrying W&B init in offline mode")
                init_kwargs["mode"] = "offline"
                run = wandb.init(**init_kwargs)
            except Exception as e2:
                logger.error(
                    "W&B offline init also failed: %s: %s", type(e2).__name__, e2
                )
                return None
        else:
            return None

    wandb.config.update(config, allow_val_change=True)
    logger.info(
        "W&B enabled: project=%s, entity=%s, name=%s, mode=%s",
        init_kwargs.get("project"),
        init_kwargs.get("entity"),
        init_kwargs.get("name"),
        init_kwargs.get("mode"),
    )

    return run


def _require_pyg():
    try:
        from torch_geometric.data import Data  # noqa: F401
        from torch_geometric.loader import LinkNeighborLoader  # noqa: F401
        from torch_geometric.nn import SAGEConv  # noqa: F401
        # LinkNeighborLoader -> NeighborSampler requires either pyg-lib or torch-sparse.
        # (Otherwise it will crash inside DataLoader workers.)
        try:
            import pyg_lib  # type: ignore  # noqa: F401

            return True
        except Exception:
            try:
                import torch_sparse  # type: ignore  # noqa: F401

                return True
            except Exception:
                import torch

                torch_ver = torch.__version__
                # torch.__version__ is like: 2.0.0+cu117
                wheel_tag = torch_ver
                raise RuntimeError(
                    "PyTorch Geometric neighbor sampling backend missing.\n"
                    "LinkNeighborLoader requires either 'pyg-lib' or 'torch-sparse'.\n\n"
                    "Fix (recommended): install the official PyG wheels that match your torch build:\n"
                        f"  python -m pip install pyg-lib torch-sparse torch-scatter \\\n    -f https://data.pyg.org/whl/torch-{wheel_tag}.html\n\n"
                    "If you're on a CPU-only machine, use the '+cpu' wheel tag instead.\n"
                    f"Detected torch: {torch_ver}"
                )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch Geometric is required for train_pyg.py.\n"
            "Install it matching your torch/cuda build (see https://pytorch-geometric.readthedocs.io).\n"
            f"Original import error: {type(e).__name__}: {e}"
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _to_numpy_2d_float32(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def _try_build_matrix_from_id_dict(obj: dict[Any, Any]) -> np.ndarray | None:
    if not obj:
        return None

    keys_int: list[int] = []
    values: list[np.ndarray] = []
    for k, v in obj.items():
        try:
            k_int = int(k)
        except Exception:
            return None
        keys_int.append(k_int)
        values.append(_to_numpy_2d_float32(v).reshape(-1))

    if not values:
        return None

    d = int(values[0].shape[0])
    if any(int(v.shape[0]) != d for v in values):
        return None

    n = int(max(keys_int)) + 1
    mat = np.zeros((n, d), dtype=np.float32)
    for k_int, v in zip(keys_int, values, strict=False):
        if k_int < 0:
            continue
        mat[k_int] = v
    return mat


def load_node_feature_matrix(path: str, *, name: str, logger: logging.Logger) -> np.ndarray:
    obj = load_compressed(path)
    if obj is None:
        raise ValueError(f"{name}: failed to load features from {path} (got None)")

    if isinstance(obj, dict):
        if "v_features" in obj:
            logger.info("%s: loaded from key 'v_features' (%s)", name, path)
            return _to_numpy_2d_float32(obj["v_features"])

        for k in ("embeddings", "embs", "x", "features"):
            if k in obj:
                logger.info("%s: loaded from key '%s' (%s)", name, k, path)
                return _to_numpy_2d_float32(obj[k])

        mat = _try_build_matrix_from_id_dict(obj)
        if mat is not None:
            logger.info(
                "%s: loaded from id->vector dict (%s) | shape=%s",
                name,
                path,
                tuple(mat.shape),
            )
            return mat

        raise ValueError(
            f"{name}: unsupported feature file format at {path}. "
            f"Expected key 'v_features' (or one of embeddings/embs/x/features) or an id->vector dict. "
            f"Keys found: {sorted(map(str, obj.keys()))[:50]}"
        )

    logger.info("%s: loaded from raw array (%s)", name, path)
    return _to_numpy_2d_float32(obj)


def _parse_config_str(value: str | None, defaults: dict[str, Any]) -> dict[str, Any]:
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


def build_edge_index_for_year(graph: Graph, year: int, num_nodes: int) -> torch.Tensor:
    # edges: (m, 3) with columns [u, v, day_offset]
    edges = graph.get_until_year(year)
    uv = edges[:, :2].astype(np.int64, copy=False)
    # undirected message passing
    uv_rev = uv[:, [1, 0]]
    uv2 = np.vstack([uv, uv_rev])
    edge_index = torch.from_numpy(uv2.T).contiguous()

    # Defensive checks
    if edge_index.numel() > 0:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise ValueError(
                f"edge_index has nodes outside [0,{num_nodes-1}]: "
                f"min={int(edge_index.min())}, max={int(edge_index.max())}"
            )

    return edge_index


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        _require_pyg()
        from torch_geometric.nn import SAGEConv

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class EdgeMLPDecoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_u, z_v, torch.abs(z_u - z_v), z_u * z_v], dim=1)
        return self.mlp(x).squeeze(1)


class DotDecoder(nn.Module):
    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        return (z_u * z_v).sum(dim=1)


def main(
    graph_path="data-v2/graph/edges.M.pkl",
    data_path="data-v2/model/data.M.pkl",
    v_features_path="data-v2/model/baseline/features.2016.binary.M.pkl.gz",
    year_start_train=2016,
    train=None,
    model=None,
    sampling=None,
    features=None,
    wandb=None,
    ood_data_path=None,
    ood_year_start=None,
    ood_eval_interval=None,
    ood_eval_batch_size=None,
    seed=42,
    log_file="logs-v2/gnn/gnn_train_pyg.log",
    save_model_path=None,
):
    """Fast GNN trainer using PyTorch Geometric neighbor sampling.

    This is intended to be a drop-in faster alternative to train.py:
    - uses LinkNeighborLoader for neighbor sampling in C++/CUDA friendly pipeline
    - uses SAGEConv layers
    - supports MLP decoder for link prediction
    """

    _require_pyg()
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader

    reload(logging)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger = setup_logger(file=log_file, level=logging.INFO, log_to_stdout=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_cfg = _parse_config_str(
        train,
        defaults={
            "batch_size": 4096,
            "pos_ratio": 0.3,
            "num_epochs": 10,
            "lr": 3e-4,
            "weight_decay": 0.0,
            "log_interval": 1,
            "eval_batch_size": 16384,
            "ood_eval_interval": 0,
            "num_workers": 8,
            "amp": True,
            "grad_clip_norm": 1.0,
        },
    )
    model_cfg = _parse_config_str(
        model,
        defaults={
            "hidden_dim": 128,
            "out_dim": 128,
            "dropout": 0.1,
            # Keep compatibility with train.py
            "decoder": "mlp",  # mlp | dot
            "decoder_hidden_dim": 256,
            "decoder_dropout": 0.1,
        },
    )
    sampling_cfg = _parse_config_str(
        sampling,
        defaults={
            "fanout1": 15,
            "fanout2": 10,
        },
    )
    features_cfg = _parse_config_str(
        features,
        defaults={
            "log1p": True,
            "zscore": True,
            "eps": 1e-6,
        },
    )

    wandb_cfg = _parse_config_str(
        wandb,
        defaults={
            "enabled": False,
            "project": "materials_concepts",
            "entity": "",
            "name": "",
            "group": "",
            "job_type": "train_pyg",
            "tags": "",
            "mode": "online",  # online | offline | disabled
            "watch": False,
            "log_model": False,
            "fail_fast": False,
        },
    )

    logger.info(f"device: {device}")
    logger.info(f"seed: {seed}")
    logger.info(f"year_start_train: {year_start_train}")
    logger.info(f"fanout: ({sampling_cfg['fanout1']}, {sampling_cfg['fanout2']})")

    wandb_run = _maybe_init_wandb(
        wandb_cfg,
        config={
            "graph_path": graph_path,
            "data_path": data_path,
            "v_features_path": v_features_path,
            "year_start_train": int(year_start_train),
            "seed": int(seed),
            "train": dict(train_cfg),
            "model": dict(model_cfg),
            "sampling": dict(sampling_cfg),
            "features": dict(features_cfg),
        },
        logger=logger,
    )

    logger.info("Loading dataset")
    data_dict = load_pickle(data_path)
    x_train = np.asarray(data_dict["X_train"], dtype=np.int64)
    y_train = np.asarray(data_dict["y_train"], dtype=np.float32)
    x_val = np.asarray(data_dict.get("X_val", data_dict.get("X_test")), dtype=np.int64)
    y_val = np.asarray(data_dict.get("y_val", data_dict.get("y_test")), dtype=np.float32)

    logger.info("Loading node features")
    v_features = load_node_feature_matrix(
        v_features_path, name="v_features", logger=logger
    )

    if bool(features_cfg.get("log1p", True)):
        if float(np.min(v_features)) < 0.0:
            logger.warning(
                "v_features contains negative values; skipping log1p transform"
            )
        else:
            v_features = np.log1p(v_features)
    if bool(features_cfg.get("zscore", True)):
        mean = v_features.mean(axis=0, keepdims=True)
        std = v_features.std(axis=0, keepdims=True)
        eps = float(features_cfg.get("eps", 1e-6))
        v_features = (v_features - mean) / (std + eps)

    num_nodes = int(v_features.shape[0])
    in_dim = int(v_features.shape[1])

    logger.info("Building past-graph edge_index")
    graph = Graph(graph_path)
    edge_index = build_edge_index_for_year(graph, year_start_train, num_nodes=num_nodes)

    if wandb_run is not None:
        try:
            wandb_run.log(
                {
                    "data/num_nodes": int(num_nodes),
                    "data/in_dim": int(in_dim),
                    "data/num_edges_undirected": int(edge_index.shape[1]),
                },
                step=0,
            )
        except Exception:
            pass

    pyg_data = Data(
        x=torch.from_numpy(v_features),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )

    # edge labels: use provided pairs (already non-edges in past graph)
    train_edge_label_index = torch.from_numpy(x_train.T).contiguous()
    train_edge_label = torch.from_numpy(y_train).to(torch.float32)

    val_edge_label_index = torch.from_numpy(x_val.T).contiguous()
    val_edge_label = torch.from_numpy(y_val).to(torch.float32)

    # Optional OOD evaluation dataset (e.g., generated at year_start_train + 3)
    ood_loader = None
    ood_y = None
    if ood_data_path:
        ood = load_pickle(ood_data_path)
        if "X_test" not in ood or "y_test" not in ood:
            raise ValueError(
                f"OOD data file must contain keys X_test and y_test: {ood_data_path}"
            )
        ood_x = np.asarray(ood["X_test"], dtype=np.int64)
        ood_y = np.asarray(ood["y_test"], dtype=np.float32)

        ood_year = (
            int(ood_year_start)
            if ood_year_start is not None
            else int(year_start_train) + 3
        )
        logger.info("Building OOD past-graph edge_index at year_start=%d", ood_year)
        edge_index_ood = build_edge_index_for_year(
            graph, ood_year, num_nodes=num_nodes
        )

        pyg_data_ood = Data(
            x=pyg_data.x,
            edge_index=edge_index_ood,
            num_nodes=num_nodes,
        )

        ood_edge_label_index = torch.from_numpy(ood_x.T).contiguous()
        ood_edge_label = torch.from_numpy(ood_y).to(torch.float32)

        if ood_eval_batch_size is None:
            ood_eval_batch_size = int(train_cfg["eval_batch_size"])

        ood_loader = LinkNeighborLoader(
            pyg_data_ood,
            edge_label_index=ood_edge_label_index,
            edge_label=ood_edge_label,
            num_neighbors=[
                int(sampling_cfg["fanout1"]),
                int(sampling_cfg["fanout2"]),
            ],
            batch_size=int(ood_eval_batch_size),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=True,
            persistent_workers=int(train_cfg["num_workers"]) > 0,
        )

        logger.info(
            "OOD pairs: %d | pos_rate: %.4f | year_start=%d",
            int(ood_x.shape[0]),
            float(ood_y.mean()) if ood_y.size else float("nan"),
            int(ood_year),
        )

    # NOTE: LinkNeighborLoader does not support "pos_ratio" directly when you provide labels.
    # If you want a controlled pos ratio, you should subsample x_train/y_train offline.
    loader = LinkNeighborLoader(
        pyg_data,
        edge_label_index=train_edge_label_index,
        edge_label=train_edge_label,
        num_neighbors=[int(sampling_cfg["fanout1"]), int(sampling_cfg["fanout2"])],
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        persistent_workers=int(train_cfg["num_workers"]) > 0,
    )

    val_loader = LinkNeighborLoader(
        pyg_data,
        edge_label_index=val_edge_label_index,
        edge_label=val_edge_label,
        num_neighbors=[int(sampling_cfg["fanout1"]), int(sampling_cfg["fanout2"])],
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        persistent_workers=int(train_cfg["num_workers"]) > 0,
    )

    encoder = SAGEEncoder(
        in_dim=in_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    decoder_kind = str(model_cfg.get("decoder", "mlp")).lower()
    if decoder_kind not in {"mlp", "dot"}:
        raise ValueError("model.decoder must be one of: mlp, dot")

    if decoder_kind == "mlp":
        decoder = EdgeMLPDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
        decoder_params = list(decoder.parameters())
        decoder_desc = f"EdgeMLPDecoder(hidden={model_cfg['decoder_hidden_dim']})"
    else:
        decoder = DotDecoder().to(device)
        decoder_params = []
        decoder_desc = "DotDecoder"

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + decoder_params,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)))

    logger.info(
        "Model: SAGEConv2Layer(in_dim=%d, hidden_dim=%s, out_dim=%s) + %s",
        in_dim,
        model_cfg["hidden_dim"],
        model_cfg["out_dim"],
        decoder_desc,
    )
    logger.info(f"Train pairs: {len(x_train)} | Val pairs: {len(x_val)}")

    if wandb_run is not None and bool(wandb_cfg.get("watch", False)):
        # Optional and can be expensive.
        try:
            import wandb  # type: ignore

            wandb.watch([encoder, decoder], log="all", log_freq=200)
        except Exception:
            pass

    if wandb_run is not None:
        try:
            n_params = int(
                sum(p.numel() for p in encoder.parameters())
                + sum(p.numel() for p in decoder.parameters())
            )
            wandb_run.log(
                {
                    "model/params": n_params,
                    "data/train_pairs": int(len(x_train)),
                    "data/val_pairs": int(len(x_val)),
                },
                step=0,
            )
        except Exception:
            pass

    def evaluate() -> tuple[float, tuple[int, int, int, int]]:
        encoder.eval()
        decoder.eval()
        scores: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Eval", leave=False):
                batch = batch.to(device, non_blocking=True)
                z = encoder(batch.x, batch.edge_index)
                u = batch.edge_label_index[0]
                v = batch.edge_label_index[1]
                logits = decoder(z[u], z[v])
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                scores.extend(probs.tolist())

        predictions = np.asarray(scores)
        auc, _, cm = test(torch.tensor(y_val, dtype=torch.float32), predictions, threshold=0.5)
        return float(auc), (int(cm[0]), int(cm[1]), int(cm[2]), int(cm[3]))

    def evaluate_ood() -> tuple[float, tuple[int, int, int, int]]:
        if ood_loader is None or ood_y is None:
            raise RuntimeError("OOD loader not initialized")
        encoder.eval()
        decoder.eval()
        scores: list[float] = []
        with torch.no_grad():
            for batch in tqdm(ood_loader, desc="OOD Eval", leave=False):
                batch = batch.to(device, non_blocking=True)
                z = encoder(batch.x, batch.edge_index)
                u = batch.edge_label_index[0]
                v = batch.edge_label_index[1]
                logits = decoder(z[u], z[v])
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                scores.extend(probs.tolist())

        predictions = np.asarray(scores)
        auc, _, cm = test(
            torch.tensor(ood_y, dtype=torch.float32), predictions, threshold=0.5
        )
        return float(auc), (int(cm[0]), int(cm[1]), int(cm[2]), int(cm[3]))

    for epoch in range(1, int(train_cfg["num_epochs"]) + 1):
        encoder.train()
        decoder.train()
        losses: list[float] = []

        for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(train_cfg.get("amp", True))):
                z = encoder(batch.x, batch.edge_index)
                u = batch.edge_label_index[0]
                v = batch.edge_label_index[1]
                logits = decoder(z[u], z[v])
                loss = criterion(logits, batch.edge_label)

            scaler.scale(loss).backward()

            grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + decoder_params,
                    max_norm=grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu().item()))

        if epoch % int(train_cfg["log_interval"]) == 0:
            auc, (tn, fp, fn, tp) = evaluate()
            logger.info(
                "Epoch: %d, Loss: %.4f, AUC: %.4f, TP: %d, FP: %d, FN: %d, TN: %d",
                epoch,
                float(np.mean(losses)) if losses else float("nan"),
                auc,
                tp,
                fp,
                fn,
                tn,
            )

            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            "epoch": int(epoch),
                            "train/loss": float(np.mean(losses))
                            if losses
                            else float("nan"),
                            "val/auc": float(auc),
                            "val/tp": int(tp),
                            "val/fp": int(fp),
                            "val/fn": int(fn),
                            "val/tn": int(tn),
                        },
                        step=int(epoch),
                    )
                except Exception:
                    pass

        # OOD eval can be expensive; default is off unless configured.
        interval = int(ood_eval_interval) if ood_eval_interval is not None else int(
            train_cfg.get("ood_eval_interval", 0)
        )
        if ood_loader is not None and interval and interval > 0 and epoch % interval == 0:
            ood_auc, (ood_tn, ood_fp, ood_fn, ood_tp) = evaluate_ood()
            logger.info(
                "OOD | Epoch: %d, AUC: %.4f, TP: %d, FP: %d, FN: %d, TN: %d",
                epoch,
                float(ood_auc),
                int(ood_tp),
                int(ood_fp),
                int(ood_fn),
                int(ood_tn),
            )

            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            "epoch": int(epoch),
                            "ood/auc": float(ood_auc),
                            "ood/tp": int(ood_tp),
                            "ood/fp": int(ood_fp),
                            "ood/fn": int(ood_fn),
                            "ood/tn": int(ood_tn),
                        },
                        step=int(epoch),
                    )
                except Exception:
                    pass

    if save_model_path:
        logger.info(f"Saving model to {save_model_path}")
        os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "year_start_train": int(year_start_train),
            },
            save_model_path,
        )

        if wandb_run is not None and bool(wandb_cfg.get("log_model", False)):
            try:
                import wandb  # type: ignore

                artifact = wandb.Artifact(
                    name=str(wandb_cfg.get("artifact_name", "train_pyg_model")),
                    type="model",
                )
                artifact.add_file(save_model_path)
                wandb_run.log_artifact(artifact)
            except Exception:
                pass

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    fire.Fire(main)
