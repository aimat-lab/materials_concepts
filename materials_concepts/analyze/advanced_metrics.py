import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from materials_concepts.utils.utils import load_pickle, load_compressed
from materials_concepts.model.graph import Graph


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _clean_scores(y_score: np.ndarray) -> np.ndarray:
    y_score = np.asarray(y_score, dtype=float)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    # If scores look like logits outside [0,1], clip to [0,1] for probabilistic metrics
    return np.clip(y_score, 0.0, 1.0)


def _stable_descending_indices(scores: np.ndarray) -> np.ndarray:
    # Stable mergesort ensures ties are deterministic by index
    return np.argsort(-scores, kind="mergesort")


def compute_pr_ap(y_true: np.ndarray, y_score: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    prevalence = float(np.mean(y_true))
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "average_precision": float(ap),
        "prevalence": prevalence,
    }


def compute_pk_rk(y_true: np.ndarray, y_score: np.ndarray, ks: List[int]):
    idx = _stable_descending_indices(y_score)
    y_sorted = y_true[idx]
    total_pos = int(y_true.sum())
    res = {}
    for k in ks:
        k_eff = min(k, len(y_sorted))
        tp_at_k = int(y_sorted[:k_eff].sum())
        precision_k = tp_at_k / k_eff if k_eff > 0 else 0.0
        recall_k = tp_at_k / total_pos if total_pos > 0 else 0.0
        res[int(k)] = {
            "k_eff": k_eff,
            "tp_at_k": tp_at_k,
            "precision@k": precision_k,
            "recall@k": recall_k,
        }
    return res


def compute_yearwise_topk(
    y_true: np.ndarray, years: np.ndarray, y_score: np.ndarray, ks: List[int]
):
    """
    Per event-year stats from a single global ranking:
    - yield_y@k: number of positives with event_year=y in top-k
    - recall_y@k: yield_y@k / total_pos_y
    - precision_y_share@k: yield_y@k / k_eff (share of top-k belonging to year y)
    """
    idx = _stable_descending_indices(y_score)
    y_sorted = y_true[idx]
    years_sorted = years[idx]

    pos_years = years[y_true.astype(bool)]
    unique_years = sorted(int(y) for y in np.unique(pos_years) if y >= 0)

    totals = {int(y): int(np.sum((y_true == 1) & (years == y))) for y in unique_years}

    out: Dict[int, Dict[int, Dict[str, float]]] = {int(y): {} for y in unique_years}

    for k in ks:
        k_eff = min(k, len(y_sorted))
        topk_years = years_sorted[:k_eff]
        topk_pos = y_sorted[:k_eff]
        # Count per-year true positives in top-k
        year_counts = Counter(int(yr) for yr in topk_years[(topk_pos == 1)])
        for y in unique_years:
            tp_y_at_k = int(year_counts.get(int(y), 0))
            total_y = totals[int(y)]
            out[int(y)][int(k)] = {
                "k_eff": k_eff,
                "yield_y@k": tp_y_at_k,
                "recall_y@k": (tp_y_at_k / total_y) if total_y > 0 else 0.0,
                "precision_y_share@k": (tp_y_at_k / k_eff) if k_eff > 0 else 0.0,
                "total_pos_y": total_y,
            }

    return out, unique_years


def compute_calibration(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_score)
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "brier_score": float(brier),
        "n_bins": int(n_bins),
    }


def plot_pr_curve(pr: Dict, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.plot(pr["recall"], pr["precision"], label=f"PR (AP={pr['average_precision']:.3f})")
    # Baseline prevalence line
    plt.hlines(pr["prevalence"], 0, 1, colors="gray", linestyles="dashed", label=f"prevalence={pr['prevalence']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_calibration(cal: Dict, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    plt.plot(cal["prob_pred"], cal["prob_true"], marker="o", label="reliability")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration (Brier={cal['brier_score']:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_hit_at_k_per_source(
    pairs: np.ndarray, y_true: np.ndarray, y_score: np.ndarray, ks: List[int]
):
    """
    Treat first vertex as the source. For each source, rank its candidate targets by score;
    Hit@k = fraction of sources with at least one positive in top-k.
    """
    u = np.asarray([p[0] for p in pairs])
    res = {}
    for k in ks:
        hits = 0
        sources = 0
        for src in np.unique(u):
            mask = (u == src)
            if mask.sum() == 0:
                continue
            sources += 1
            idx = _stable_descending_indices(y_score[mask])
            y_sorted = y_true[mask][idx]
            k_eff = min(k, len(y_sorted))
            if int(y_sorted[:k_eff].sum()) > 0:
                hits += 1
        res[int(k)] = {
            "sources": int(sources),
            "hit_sources": int(hits),
            "hit@k": (hits / sources) if sources > 0 else 0.0,
        }
    return res


def main(
    data_path: str = "data-v2/model/val.data.M.pkl",
    predictions_path: str = "data-v2/model/combi/predictions.pkl.gz",
    out_dir: str = "data-v2/model/combi/analysis",
    dist_path: str = "data-v2/graph/unwrapped_diststances.pkl",
    ks: str = "10,100,1000,10000,100000",
    n_bins: int = 10,
    compute_per_source_hit: bool = False,
    # Years computation options when years_test missing
    graph_path: Optional[str] = "data-v2/graph/edges.M.pkl",
    start_year: Optional[int] = 2019,
    end_year: Optional[int] = 2022,
    years_cache_path: Optional[str] = "data-v2/model/years.M.pkl",
):
    """
    Compute reviewer-requested metrics and plots for a single global cohort with event years.

    Expects data dict with keys:
      - X_test: list/array of (u, v)
      - y_test: 0/1 labels
      - years_test: event year for positives, -1 for negatives
    Predictions should be probabilities in [0,1] aligned with y_test.
    """
    _ensure_dir(out_dir)

    data = load_pickle(data_path)
    y_true = np.asarray(data["y_test"]).astype(int)
    # Years: if missing, compute via Graph
    if "years_test" in data and data["years_test"] is not None:
        years = np.asarray(data["years_test"]).astype(int)
    else:
        years = None
        # Optimization: check for cache before loading the (potentially large) graph
        if years_cache_path:
            try:
                cached_data = load_pickle(years_cache_path)
                if isinstance(cached_data, dict) and "years" in cached_data:
                    print(f"Found and loaded cached event years from {years_cache_path}")
                    years = np.asarray(cached_data["years"], dtype=int)
                elif isinstance(cached_data, (list, np.ndarray)): # Backwards compatibility
                    print(f"Found and loaded cached event years from {years_cache_path}")
                    years = np.asarray(cached_data, dtype=int)
            except FileNotFoundError:
                pass # Cache not found, will compute now

        if years is None:
            if graph_path is None or start_year is None:
                raise ValueError("years_test missing: provide graph_path and start_year to compute event years")
            print("years_test missing and no valid cache found; computing event years from graph...")
            g = Graph.from_path(graph_path)
            years = g.compute_event_years_for_pairs(
                np.asarray(data["X_test"]),
                start_year=int(start_year),
                end_year=int(end_year) if end_year is not None else None,
                cache_path=years_cache_path,
            )
    pairs = np.asarray(data["X_test"]) if "X_test" in data else None

    y_score_raw = load_compressed(predictions_path)
    y_score = _clean_scores(np.asarray(y_score_raw))

    # Debugging check for year assignment
    pos_mask = y_true == 1
    pos_years = years[pos_mask]
    missing_pos_years = np.sum(pos_years == -1)
    if missing_pos_years > 0:
        print(
            f"WARNING: Found {missing_pos_years} positive labels (out of {np.sum(pos_mask)}) "
            f"with an event year of -1. These will be excluded from year-wise metrics. "
            f"Check if `start_year` and `end_year` are correct and if the graph data is complete."
        )

    dists = np.asarray(load_pickle(dist_path))
    if dists.ndim != 1:
        dists = np.asarray(dists).reshape(-1)

    if len(dists) != len(y_true):
        raise ValueError(
            f"Distance array length {len(dists)} does not match number of labels {len(y_true)}"
        )

    dist_mask = dists == 3
    if not np.any(dist_mask):
        raise ValueError("No samples found at distance 3; check distance data and path")

    y_true = y_true[dist_mask]
    y_score = y_score[dist_mask]
    years = years[dist_mask]
    if pairs is not None:
        pairs = pairs[dist_mask]

    print("Filtered to distance==3:", y_true.shape, y_score.shape, years.shape)

    assert len(y_true) == len(y_score), "Length of predictions must match y_test"
    assert len(years) == len(y_true), "Length of years_test must match y_test"

    ks_list = [int(k.strip()) for k in ks.split(",") if k.strip()]

    # Global metrics
    pr = compute_pr_ap(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    pk_rk = compute_pk_rk(y_true, y_score, ks_list)
    cal = compute_calibration(y_true, y_score, n_bins=n_bins)

    # Year-wise metrics
    yearwise, unique_years = compute_yearwise_topk(y_true, years, y_score, ks_list)

    # Optional per-source Hit@k
    per_source_hit = None
    if compute_per_source_hit and pairs is not None:
        per_source_hit = compute_hit_at_k_per_source(pairs, y_true, y_score, ks_list)

    # Save metrics JSON
    metrics = {
        "global": {
            "auc": auc,
            "average_precision": pr["average_precision"],
            "prevalence": pr["prevalence"],
            "precision_at_k": pk_rk,
            "brier_score": cal["brier_score"],
        },
        "yearwise": yearwise,
        "years": unique_years,
    }
    if per_source_hit is not None:
        metrics["per_source_hit@k"] = per_source_hit

    with open(os.path.join(out_dir, "dist_3_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics and plots to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
