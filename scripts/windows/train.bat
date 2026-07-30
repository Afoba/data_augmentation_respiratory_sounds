rmdir /s /q "results"

python src/training/train.py --config config/experiments/original.yaml
python src/training/train.py --config config/experiments/gauss_noise.yaml
python src/training/train.py --config config/experiments/pitch_shift.yaml
python src/training/train.py --config config/experiments/time_stretch.yaml
python src/training/visualize_results.py --run-name paper
