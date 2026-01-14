# GNN (Topology-only) link prediction

This folder contains an MVP graph neural network (GNN) trainer for the same temporal link prediction setup used by the existing MLP baseline.

## What problem this solves

We want to predict whether a pair of nodes `(u, v)` that is **not connected at** `year_start_train` will become connected **within the next 3 years** (as encoded by the labels in `data/model/data.pkl`).

This matches the data generation logic in `materials_concepts/model/create_data.py`:
- positives: edges that appear in the *future* graph but were not present in the *past* graph
- negatives: pairs that are non-edges in both past and future

## How the implemented GNN mirrors the existing baseline

The current MLP baseline learns a classifier over **pair features** (e.g., `v_features[u] || v_features[v]`).

The implemented GNN keeps the **same supervision** (batches are still just labeled node pairs), but replaces the “static pair feature” with **message passing on the past graph**:

1. Sample a mini-batch of labeled pairs `(u, v, y)` from `X_train/y_train` (optionally with a controlled positive ratio).
2. Collect the unique endpoints `S0 = {u and v in the batch}`.
3. Build a *mini-batch computation subgraph* by sampling neighbors on the **past graph at `year_start_train`**:
   - sample up to `fanout1` 1-hop neighbors per seed
   - sample up to `fanout2` 2-hop neighbors per 1-hop node
4. Run a 2-layer GraphSAGE-style encoder to compute node embeddings `z_u, z_v` for the batch endpoints.
5. Decode a link probability using a dot-product decoder:

   `logit(u, v) = z_u · z_v` and `p = sigmoid(logit)`

6. Train with binary cross-entropy on logits (BCEWithLogitsLoss).

Evaluation uses the same `X_val/y_val` (or `X_test/y_test`) pair lists to keep AUC comparable to the baseline.

## Why this is still a “GNN”

A model is considered a GNN when node representations are computed by **neighborhood aggregation / message passing**.

GraphSAGE is a standard message-passing GNN; the only difference here is that we use **neighbor sampling** to make training feasible on large graphs.

## Why this design for a large, hub-heavy graph

Your graph has >100k nodes and millions of edges with very large hubs. Plain full-neighborhood message passing can be problematic:
- it is computationally expensive (hubs dominate)
- it can wash out signal (hub neighborhoods overwhelm local structure)

This implementation mitigates that by:
- **capping neighbor fanout** per node per layer (`fanout1`, `fanout2`)
- using a **shallow** 2-layer encoder (reduces oversmoothing)
- keeping the decoder simple (dot product), so improvements reflect the encoder rather than a heavy MLP head

## Node features used (MVP)

The MVP starts with topology-only node features `v_features` produced by `materials_concepts/model/combi/pre_compute.py`.

These features already contain multi-year structural signal; the GNN’s message passing uses only the adjacency from `year_start_train` to avoid label leakage.

## How to run

Example (defaults shown):

```bash
pixi run python materials_concepts/model/gnn/train.py \
  --graph_path=data/graph/edges.pkl \
  --data_path=data/model/data.pkl \
  --v_features_path=data/model/combi/matrices_2016.pkl.gz \
  --year_start_train=2016
```

Hyperparameters are passed as simple `key=value` comma-separated strings:

```bash
pixi run python materials_concepts/model/gnn/train.py \
  --train="batch_size=512,pos_ratio=0.3,num_epochs=50,lr=1e-3,weight_decay=0.0,log_interval=5,eval_batch_size=4096" \
  --model="hidden_dim=64,out_dim=64,dropout=0.1" \
  --sampling="fanout1=15,fanout2=10"
```

## Notes / future extensions

- Swap the decoder to an MLP over `[z_u || z_v || |z_u-z_v| || z_u*z_v]` if dot-product saturates.
- If hub effects remain strong, add edge dropout or degree-aware sampling/normalization.
- If/when text embeddings are added, keep the same flow: text vectors become node features, and supervision remains pair-based.
