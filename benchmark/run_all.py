"""
How to use:
    uv run python run_all.py configs/fasttext.yaml
"""

import argparse
import os
from pathlib import Path

import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from run import run

CONFIGS_DIR = Path(__file__).parent / "configs"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the model YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    datasets = list(cfg["datasets"].keys())
    print(f"Datasets: {datasets}")

    for dataset_name in datasets:
        run(args.config, dataset_name)

    print("\nAll benchmarks completed.")
