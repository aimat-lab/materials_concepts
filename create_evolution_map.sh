for year in {1990..2023}
do
    if [ -f "report/pdf/evolution/word-embs.${year}.M.pkl.gz" ]; then
        echo "Processing year ${year}"
        python model/reduce_dim.py --input_file "report/pdf/evolution/word-embs.${year}.M.pkl.gz" "report/pdf/evolution/transformed.${year}.pkl.gz"
    fi
done