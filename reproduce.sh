export CHUNK_SIZE=10000

echo "Reproducing test set evaluation"

mkdir -p test-data/eval

echo "BASELINE"
python -u materials_concepts/model/combi/eval.py \
  --data_path test-data/test.data.M.pkl\
  --emb_f_test_path test-data/baseline/features.2019.binary.M.pkl.gz \
  --emb_c_test_path False \
  --layers "[20, 300, 180, 108, 64, 10, 1]" \
  --dropout 0.25 \
  --model_path test-data/baseline/model.pt \
  --csv_path test-data/eval/baseline_thresholds.csv \
  --pred_path test-data/eval/baseline_predictions.pkl.gz \
  --metrics_path test-data/eval/baseline_metrics.pkl.gz \
  --chunk_size $CHUNK_SIZE

# AUC 0.9109
# Precision 0.0018
# Recall 0.6938
# F1 0.0035
# Confusion matrix:
# TN: 1878544, FP: 121149, FN: 94, TP: 213

echo ""
echo "PURE_EMBS"
python -u materials_concepts/model/combi/eval.py \
  --data_path test-data/test.data.M.pkl\
  --emb_f_test_path False \
  --emb_c_test_path test-data/pure_embs/features.concept-embs.2019.M.pkl.gz \
  --layers "[1536, 1024, 819, 10, 1]" \
  --dropout 0.1 \
  --model_path test-data/pure_embs/model.pt \
  --csv_path test-data/eval/pure_embs_thresholds.csv \
  --pred_path test-data/eval/pure_embs_predictions.pkl.gz \
  --metrics_path test-data/eval/pure_embs_metrics.pkl.gz \
  --chunk_size $CHUNK_SIZE

# AUC 0.8855
# Precision 0.0014
# Recall 0.6808
# F1 0.0028
# Confusion matrix:
# TN: 1852882, FP: 146811, FN: 98, TP: 209

echo ""
echo "COMBI"
python -u materials_concepts/model/combi/eval.py \
  --data_path test-data/test.data.M.pkl\
  --emb_f_test_path test-data/baseline/features.2019.binary.M.pkl.gz \
  --emb_c_test_path test-data/pure_embs/features.concept-embs.2019.M.pkl.gz \
  --layers "[1556, 1556, 933, 559, 335, 10, 1]" \
  --dropout 0.1 \
  --model_path test-data/combi/model.pt \
  --csv_path test-data/eval/combi_thresholds.csv \
  --pred_path test-data/eval/combi_predictions.pkl.gz \
  --metrics_path test-data/eval/combi_metrics.pkl.gz \
  --chunk_size $CHUNK_SIZE

echo ""
echo "Mixture"
# to avoid re-running the baseline and pure_embs models, we can just use the predictions from the previous runs and blend them

python -u materials_concepts/model/mixture/blend.py \
  --data_path test-data/test.data.M.pkl \
  --predictions_path_1 "test-data/eval/baseline_predictions.pkl.gz" \
  --predictions_path_2 "test-data/eval/pure_embs_predictions.pkl.gz" \
  --save_path "test-data/eval/mixture_predictions.pkl.gz" \
  --metrics_path "test-data/eval/mixture_metrics.pkl.gz" \
  --blend "[0.6, 0.4]" 
