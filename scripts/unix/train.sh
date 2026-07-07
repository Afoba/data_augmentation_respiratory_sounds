#!/bin/sh

rm -rf "results/"

python3 src/training/train.py --config config/experiments/paper.yaml
python3 src/training/visualize_results.py --run-name paper