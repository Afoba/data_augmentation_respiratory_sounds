#!/bin/sh

#SBATCH --job-name=train
#SBATCH --partition=normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4           # HF model loading is CPU-heavy at startup
#SBATCH --mem=16G                    # 16-28 GB model + game env headroom

python3 src/training/train.py --config config/experiments/original.yaml
python3 src/training/visualize_results.py --run-name original
