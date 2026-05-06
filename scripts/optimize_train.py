from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.ml.optimization import OptimizationConfig, run_optimization


DEFAULT_SPLIT_CSV = "data/splits/dataset_lsc70w.csv"
DEFAULT_MODEL_TYPE = "svm"
DEFAULT_TRIALS = 30
DEFAULT_SEED = 1337


def load_split_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_features: List[List[float]] = []
    train_labels: List[str] = []
    val_features: List[List[float]] = []
    val_labels: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("split CSV must include split and label columns")

        feature_columns = [
            column
            for column in reader.fieldnames
            if column not in {"split", "label", "filepath", "participant"}
        ]
        if not feature_columns:
            raise ValueError("split CSV must include numeric feature columns for optimization")

        for row in reader:
            if row["split"] not in {"train", "val"}:
                continue
            vector = [float(row[column]) for column in feature_columns]
            if row["split"] == "train":
                train_labels.append(row["label"])
                train_features.append(vector)
            else:
                val_labels.append(row["label"])
                val_features.append(vector)

    if not train_features:
        raise ValueError("no training rows found in split CSV")
    if not val_features:
        raise ValueError("no validation rows found in split CSV")

    return (
        np.asarray(train_features, dtype=np.float32),
        np.asarray(train_labels),
        np.asarray(val_features, dtype=np.float32),
        np.asarray(val_labels),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize hyperparameters and train best model")
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--model-type", choices=["svm", "mlp"], default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default="artifacts/experiments/hyperparam_optimization")
    parser.add_argument("--registry-dir", default="models/registry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    train_features, train_labels, val_features, val_labels = load_split_data(Path(args.split_csv))
    config = OptimizationConfig(
        model_type=args.model_type,
        n_trials=args.n_trials,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        registry_dir=Path(args.registry_dir),
    )

    study, artifacts = run_optimization(
        train_features,
        train_labels,
        config,
        validation_features=val_features,
        validation_labels=val_labels,
    )
    logging.info("Best score: %.4f", study.best_value)
    logging.info("Best params: %s", artifacts["best_params"])
    logging.info("Best model: %s", artifacts["model"])


if __name__ == "__main__":
    main()
