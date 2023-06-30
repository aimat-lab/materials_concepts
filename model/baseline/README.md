## Train / Eval model

```
python model/baseline/train.py \
  --data_path model/data.pkl \
  --embeddings_path model/baseline/embeddings.pkl \
  --lr 0.001 \
  --batch_size 100 \
  --num_epochs 1 \
  --train_model True \
  --save_model model/baseline/model.pt
```

```
python model/baseline/train.py \
  --data_path=model/data.pkl \
  --embeddings_path=model/baseline/embeddings.pkl \
  --lr=0.001 \
  --batch_size=100 \
  --num_epochs=1 \
  --train_model=True \
  --save_model=model/baseline/model.pt
```
