## Preparation

- [ ] Generate training data
- [ ] Training/Test Split 0.9 / 0.1 (real probs)
- [ ] For 10.000.000 Vertex Pairs <= 10
- [ ] Undersample training data
- [ ] Pickle data

Input

```
def create_training_data(
    full_graph,
    year_start,
    years_delta: (1,3,5),
    edges_used=500_000,
    c (0,5,25)
    w (1,3)
)

def create_test_data -> real probabilities, 10.000.000 vertex pairs, only if
degree >= 10?
```

Output

```
{
    'year': 2017,
    'delta': 3,

    'X_train': X_train,     # vertex pairs  [(13,62), (402,83), ...]
    'y_train': y_train,     # connected     [0,1,0,0,...]
    'X_test': X_test,
    'y_test': y_test
}
```

## The Task

Given 1.000.000 vertex pairs (m) embedded in their graph (g) until year 2017 (y), decide whether they are connected or not in next 3 years.

Consult output.pkl (m, X_train, y_train) and edges.npz (graph). Slice graph according to year. Generate embeddings for vertex pairs. Train model on embeddings (again 0.9:0.1 split on training data).

## Evaluation

Evaluate on real data (X_test, y_test).

- [ ] Plot confusion matrix
