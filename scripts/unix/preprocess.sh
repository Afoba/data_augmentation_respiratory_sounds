#!/bin/sh

#SBATCH --job-name=train
#SBATCH --partition=normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4           # HF model loading is CPU-heavy at startup
#SBATCH --mem=16G                    # 16-28 GB model + game env headroom

rm -rf "data/processed/features"
rm -f "data/processed/metadata.parquet"

python3 src/preprocessing/build_features.py --datasets icbhi sprsound --features mel --workers 4
# python3 src/preprocessing/build_augmented.py --shifts -2 -1 1 2 --features logmel --workers 4
python3 src/preprocessing/build_val_split.py --val-fraction 0.1 --seed 42