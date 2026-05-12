from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.ml.feature_extraction import FeatureConfig, extract_features
from src.ml.preprocessing import Preprocessor
from src.ml.training import evaluate_model, persist_artifacts, train_mlp, train_svm


# === Constants ===
DEFAULT_SPLIT_CSV = "data/splits/dataset_lsc70anh_abcde.csv"
DEFAULT_MODEL_TYPE = "svm"
DEFAULT_SEED = 1337
DEFAULT_OUTPUT_DIR = "artifacts/experiments/direct_training"
DEFAULT_REGISTRY_DIR = "models/registry"
DEFAULT_SVM_C = 1.0
DEFAULT_SVM_LOSS = "squared_hinge"
DEFAULT_MLP_HIDDEN_LAYER_SIZES = (64,)
DEFAULT_MLP_ALPHA = 0.0001
DEFAULT_MLP_LEARNING_RATE_INIT = 0.001
REQUIRED_MANIFEST_COLUMNS = {"filepath", "label", "split"}


# === Data Loading ===
def resolve_manifest_path(filepath: str, csv_path: Path) -> Path:
    """Resolve manifest filepath to absolute path."""
    path = Path(filepath)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return csv_path.parent / path


def load_split_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess data from a split CSV.
    
    Returns:
        Tuple of (train_features, train_labels, val_features, val_labels)
    """
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
    # disable=not sys.stdout.isatty() automatically suppresses progress bars in non-interactive contexts
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


# === CLI Argument Parsing ===
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for direct model training."""
    parser = argparse.ArgumentParser(
        description="Train a model directly with specified hyperparameters (no optimization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SVM with defaults
  python scripts/train.py --split-csv data/splits/dataset_lsc70anh_abcde.csv --model svm
  
  # Train SVM with custom hyperparameters
  python scripts/train.py --model svm --C 10 --loss "squared_hinge"
  
  # Train MLP with defaults
  python scripts/train.py --model mlp
  
  # Train MLP with custom hyperparameters
  python scripts/train.py --model mlp --hidden-layer-sizes 128 64 --alpha 0.001 --learning-rate-init 0.002
        """,
    )

    # Common arguments
    parser.add_argument(
        "--split-csv",
        default=DEFAULT_SPLIT_CSV,
        help=f"Path to split CSV with filepath, label, split columns (default: {DEFAULT_SPLIT_CSV})",
    )
    parser.add_argument(
        "--model",
        choices=["svm", "mlp"],
        default=DEFAULT_MODEL_TYPE,
        help=f"Model type to train (default: {DEFAULT_MODEL_TYPE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for training logs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--registry-dir",
        default=DEFAULT_REGISTRY_DIR,
        help=f"Model registry directory (default: {DEFAULT_REGISTRY_DIR})",
    )

    # SVM-specific arguments
    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_SVM_C,
        help=f"SVM regularization parameter (default: {DEFAULT_SVM_C})",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=DEFAULT_SVM_LOSS,
        help=f"SVM loss function (default: {DEFAULT_SVM_LOSS})",
    )

    # MLP-specific arguments
    parser.add_argument(
        "--hidden-layer-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_MLP_HIDDEN_LAYER_SIZES),
        help=f"MLP hidden layer sizes (default: {DEFAULT_MLP_HIDDEN_LAYER_SIZES[0]})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_MLP_ALPHA,
        help=f"MLP L2 regularization parameter (default: {DEFAULT_MLP_ALPHA})",
    )
    parser.add_argument(
        "--learning-rate-init",
        type=float,
        default=DEFAULT_MLP_LEARNING_RATE_INIT,
        help=f"MLP initial learning rate (default: {DEFAULT_MLP_LEARNING_RATE_INIT})",
    )

    return parser.parse_args()


# === Main Training Function ===
def main() -> None:
    """Main entry point for direct model training."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("direct_training")

    try:
        # Log training start
        logger.info("Starting %s training with hyperparameters", args.model)
        if args.model == "svm":
            logger.info("  C=%.4f, loss=%s", args.C, args.loss)
        else:  # mlp
            logger.info("  hidden_layer_sizes=%s, alpha=%.6f, learning_rate_init=%.6f",
                       tuple(args.hidden_layer_sizes), args.alpha, args.learning_rate_init)

        # Load data
        logger.info("Loading and preprocessing data from %s", args.split_csv)
        train_features, train_labels, val_features, val_labels = load_split_data(Path(args.split_csv))
        logger.info("Data loaded: %d training samples, %d validation samples",
                   train_features.shape[0], val_features.shape[0])

        # Train model
        logger.info("Training %s model", args.model)
        if args.model == "svm":
            model = train_svm(
                train_features,
                train_labels,
                seed=args.seed,
                C=args.C,
                loss=args.loss,
            )
        else:  # mlp
            model = train_mlp(
                train_features,
                train_labels,
                seed=args.seed,
                hidden_layer_sizes=tuple(args.hidden_layer_sizes),
                alpha=args.alpha,
                learning_rate_init=args.learning_rate_init,
            )

        # Evaluate model
        logger.info("Evaluating model on validation data")
        metrics = evaluate_model(model, val_features, val_labels)
        f1_macro = metrics["f1_macro"]
        logger.info("Validation macro F1: %.4f", f1_macro)
        logger.info("Classification report:\n%s", metrics["classification_report"])

        # Persist model
        logger.info("Persisting model and metrics")
        run_id = f"direct-{args.model}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        artifacts = persist_artifacts(
            model,
            val_features,
            val_labels,
            Path(args.registry_dir),
            model_name=args.model,
            run_id=run_id,
        )

        logger.info("Model saved to: %s", artifacts["model"])
        logger.info("Metrics saved to: %s", artifacts["metrics"])
        logger.info("Run directory: %s", artifacts["run_dir"])
        logger.info("Training complete!")

    except Exception as e:
        logger.error("Training failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
