rmdir /s /q "results/paper"

python src/training/train.py --config config/experiments/paper.yaml
python src/training/visualize_results.py --run-name paper
