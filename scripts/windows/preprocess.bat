rmdir /s /q "data/processed/features"
del /f /q "data/processed/metadata.parquet"

python src/preprocessing/build_features.py --datasets icbhi sprsound --features mel --workers 4
:: python src/preprocessing/build_augmented.py --shifts -2 -1 1 2 --features logmel --workers 4
python src/preprocessing/build_val_split.py --val-fraction 0.1 --seed 42
