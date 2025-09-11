To reproduce the reported AUC values, make sure to download all the figshare data and arange it in a `test-data` dir (or anywhere else) like this:

```bash
test-data/
├── baseline
│   ├── auc_curve.pkl.gz
│   ├── features.2016.binary.M.pkl.gz
│   ├── features.2019.binary.M.pkl.gz # <- this will be used for eval
│   ├── features.2022.binary.M.pkl.gz # <- this is only for predictions with all model knowledge available
│   └── model.pt
├── combi
│   └── model.pt
├── mixture # <- not really needed, since it's only a blend of models
├── pure_embs
│   ├── features.concept-embs.2016.M.pkl.gz
│   ├── features.concept-embs.2019.M.pkl.gz # <- this will be used for eval
│   ├── features.concept-embs.2022.M.pkl.gz # <- this is only for predictions with all model knowledge available
│   └── model.pt
└── test.data.M.pkl
```

Run the `reproduce.sh` script that will evaluate the models on the test data. You'll need to adapt your storage layout to the script or vice versa, since figshare only allows for uploading a flattened directory.

Before running, please make sure you've followed the installation guide and your virtual env is activated. Also make sure to run the script on a machine with sufficient RAM. The inference can be carried out on a CPU, but 2 million data points to evaluate are quite many. Therefore, we emply a batched inference with a `chunk_size` of 10k, feel free to adapt that in the reproduce script. 

The metrics and predictions are saved, allowing to calculate the ROC curves as well using `sklearn.metrics.roc_curve`.

## Further Reproducibility Notes

MatSciBERT from hugging face (`m3rg-iitd/matscibert`) with commit hash: `24a4e4318dda9bc18bff5e6a45debdcb3e1780e3`.

BERT from hugging face (`bert-base-uncased`) with commit hash: `86b5e0934494bd15c9632b12f734a8a67f723594`.

Random seed for all numpy and torch operations: `42` (this is also the default setting).