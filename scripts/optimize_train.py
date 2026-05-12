from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from src.ml.feature_extraction import FeatureConfig, extract_features
from src.ml.optimization import OptimizationConfig, run_optimization
from src.ml.preprocessing import Preprocessor


DEFAULT_SPLIT_CSV = "data/splits/dataset_lsc70anh_abcde.csv"
DEFAULT_MODEL_TYPE = "svm"
DEFAULT_TRIALS = 30
DEFAULT_SEED = 1337
REQUIRED_MANIFEST_COLUMNS = {"filepath", "label", "split"}


def load_split_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = Preprocessor()
    feature_config = FeatureConfig()

    train_features: List[np.ndarray] = []
    train_labels: List[str] = []
    val_features: List[np.ndarray] = []
    val_labels: List[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_MANIFEST_COLUMNS.issubset(reader.fieldnames):
            required = ", ".join(sorted(REQUIRED_MANIFEST_COLUMNS))
            raise ValueError(f"split CSV must include manifest columns: {required}")

        rows = list(reader)

    # Wrap the feature extraction loop with tqdm for progress visualization
    # disable=not sys.stdout.isatty() automatically suppresses progress bars in non-interactive contexts (CI, redirected output)
    # desc="Preprocessing" labels the progress bar with operation context
    # unit="sample" shows semantic units (samples processed per second)
    for row_number, row in enumerate(
        tqdm(rows, desc="Preprocessing", unit="sample", disable=not sys.stdout.isatty()),
        start=2
    ):
        split = row["split"]
        if split not in {"train", "val"}:
            continue

        image_path = resolve_manifest_path(row["filepath"], csv_path)
        try:
            image = preprocessor.process_path(str(image_path))
            vector = extract_features(image, feature_config)
        except Exception as exc:
            raise OSError(
                f"could not extract features for row {row_number} "
                f"from '{row['filepath']}': {exc}"
            ) from exc

        if split == "train":
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
        np.vstack(train_features).astype(np.float32, copy=False),
        np.asarray(train_labels),
        np.vstack(val_features).astype(np.float32, copy=False),
        np.asarray(val_labels),
    )


def resolve_manifest_path(filepath: str, csv_path: Path) -> Path:
    path = Path(filepath)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return csv_path.parent / path


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
