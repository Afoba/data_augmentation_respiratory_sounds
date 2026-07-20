#!/bin/sh

#SBATCH --job-name=train
#SBATCH --partition=normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4           # HF model loading is CPU-heavy at startup
#SBATCH --mem=16G                    # 16-28 GB model + game env headroom

python3 src/preprocessing/build_features.py --datasets icbhi --features logmel
python3 src/preprocessing/build_val_split.py --val-fraction 0.1 --seed 42

python3 src/preprocessing/build_augmented.py --target-map crackle=2044 wheeze=1636 both=1706 \
    --techniques additive_gaussian_noise --generator-prefix noise_ds --workers 4

python3 src/preprocessing/build_augmented.py --target-map crackle=2044 wheeze=1636 both=1706 \
    --techniques time_stretch --generator-prefix stretch_ds --workers 4

python3 src/preprocessing/build_augmented.py --target-map crackle=2044 wheeze=1636 both=1706 \
    --techniques pitch_shift --generator-prefix pitch_ds --workers 4

python3 src/preprocessing/build_specaug.py --target-map crackle=2044 wheeze=1636 both=1706 \
    --generator-prefix offline_aug --workers 4

python3 src/preprocessing/build_balanced_specflip.py --mode vertical \
    --target 500 --exact --generator-prefix specflip_v --workers 4

python3 src/preprocessing/build_balanced_specflip.py --mode horizontal \
    --target 500 --exact --generator-prefix specflip_h --workers 4

python3 src/preprocessing/build_mixup_pairs.py --n-samples 500 \
    --generator-prefix specmixup --workers 4