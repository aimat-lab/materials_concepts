
for year in {1990..2023}
do
    if [ -f "evolution/transformed.${year}.pkl.gz" ]; then
        echo "Processing year ${year}"
        python plot_map.py --embeddings "evolution/transformed.${year}.pkl.gz" --output "evolution/${year}.png"
    fi
done