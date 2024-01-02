#!/bin/sh
#SBATCH --partition=single
#SBATCH --job-name=plot-map
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs-v2/plot-map.out
#SBATCH --mail-user=fb6372@partner.kit.edu

for year in {1990..2023}
do
    for month in {0..11}
    do
        if [ ! -f "data-v2/evolution-monthwise/image/map.${year}-${month}.pkl.gz" ]; then
            echo "Processing year ${year} month ${month}"
            python plot_map.py --embeddings "data-v2/evolution-monthwise/compressed/word-embs.${year}-${month}.pkl.gz" --output "data-v2/evolution-monthwise/image/map.${year}-${month}.png"
        fi
    done
done