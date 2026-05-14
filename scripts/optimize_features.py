# /// script
# dependencies =[
#    "optuna",
#    "numpy",
#    "scikit-learn",
#    "matplotlib",
#    "matplotlib-inline",
#    "scikit-image",
#    "opencv-python",
# ]
# ///
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import csv
import json
import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern

from src.ml.preprocessing import PreprocessConfig, Preprocessor
from src.ml.feature_extraction import FeatureConfig, HogConfig, LbpConfig, extract_features


DEFAULT_SEED = 1337
DEFAULT_DATASET_CSV = "data/processed/dataset_lsc70anh_abcde.csv"
DEFAULT_OUTPUT_DIR = "artifacts/experiments/feature_optimization"


@dataclass
class OptimizationConfig:
    n_trials: int = 30
    top_k: int = 5
    seed: int = DEFAULT_SEED
    study_name: str = "hog_lbp_feature_optimization"


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("feature_optimization")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def load_dataset(csv_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "filepath" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("dataset CSV must include 'filepath' and 'label' columns")
        for row in reader:
            rows.append((row["filepath"], row["label"]))
    if not rows:
        raise ValueError("dataset CSV contains no rows")
    return rows


def resolve_paths(rows: Iterable[Tuple[str, str]]) -> List[Tuple[Path, str]]:
    resolved: List[Tuple[Path, str]] = []
    for filepath, label in rows:
        path = Path(filepath)
        resolved.append((path, label))
    return resolved


def validate_hog_params(image_shape: Tuple[int, int], pixels_per_cell: int, cells_per_block: int) -> None:
    if pixels_per_cell <= 0 or cells_per_block <= 0:
        raise ValueError("pixels_per_cell and cells_per_block must be positive")
    height, width = image_shape
    if height % pixels_per_cell != 0 or width % pixels_per_cell != 0:
        raise ValueError("image shape must be divisible by pixels_per_cell")
    cells_y = height // pixels_per_cell
    cells_x = width // pixels_per_cell
    if cells_per_block > cells_y or cells_per_block > cells_x:
        raise ValueError("cells_per_block must not exceed available cells")


def validate_lbp_params(radius: int, n_points: int, method: str) -> None:
    if n_points != 8 * radius:
        raise ValueError("n_points must equal 8 * radius")
    if method != "uniform":
        raise ValueError("LBP method must be 'uniform'")


def preprocess_images(
    samples: Iterable[Tuple[Path, str]],
    preprocessor: Preprocessor,
    logger: logging.Logger,
) -> Tuple[List[np.ndarray], List[str], List[Tuple[Path, str]]]:
    images: List[np.ndarray] = []
    labels: List[str] = []
    kept: List[Tuple[Path, str]] = []
    for path, label in samples:
        try:
            img = preprocessor.process_path(str(path))
        except Exception as exc:
            logger.warning("Skipping corrupted image %s: %s", path, exc)
            continue
        images.append(img)
        labels.append(label)
        kept.append((path, label))
    if not images:
        raise ValueError("no valid images after preprocessing")
    return images, labels, kept


def build_feature_matrix(images: Iterable[np.ndarray], config: FeatureConfig) -> np.ndarray:
    features = [extract_features(image, config) for image in images]
    return np.vstack(features)


def evaluate_trial(
    images: List[np.ndarray],
    labels: List[str],
    hog_cfg: HogConfig,
    lbp_cfg: LbpConfig,
    seed: int,
) -> Tuple[float, str]:
    feature_config = FeatureConfig(hog=hog_cfg, lbp=lbp_cfg)
    X = build_feature_matrix(images, feature_config)
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    model = LogisticRegression(
        solver="lbfgs",
        C=0.1,
        max_iter=50,
        n_jobs=-1,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
        for warn in caught:
            if "ConvergenceWarning" in str(warn.message):
                logging.getLogger("feature_optimization").warning("%s", warn.message)

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    report = classification_report(y_val, preds, zero_division=0)
    return f1, report


def sample_params(trial: optuna.trial.Trial) -> Tuple[HogConfig, LbpConfig]:
    """
    Sample hyperparameters for HOG and LBP configurations.
    """

    radius = trial.suggest_int("lbp_radius", 1, 3)
    n_points = 8 * radius
    orientations = trial.suggest_int("hog_orientations", 6, 12)
    pixels_per_cell = trial.suggest_categorical("hog_pixels_per_cell", [4, 8, 16])
    cells_per_block = trial.suggest_categorical("hog_cells_per_block", [1, 2, 3])

    lbp_cfg = LbpConfig(radius=radius, n_points=n_points, method="uniform")
    hog_cfg = HogConfig(
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
    )
    return hog_cfg, lbp_cfg


def log_trial(log_rows: List[dict], trial: optuna.trial.FrozenTrial) -> None:
    score = trial.value if trial.value is not None else float("nan")
    log_rows.append({
        "trial": trial.number,
        "score": score,
        **trial.params,
        "state": trial.state.name,
    })


def save_trial_log(log_rows: List[dict], csv_path: Path) -> None:
    if not log_rows:
        return
    fieldnames = list(log_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)


def save_top_k(log_rows: List[dict], k: int, csv_path: Path) -> None:
    if k <= 0 or not log_rows:
        return
    sorted_rows = [row for row in log_rows if np.isfinite(row.get("score", float("nan")))]
    sorted_rows = sorted(sorted_rows, key=lambda row: row.get("score", 0.0), reverse=True)[:k]
    save_trial_log(sorted_rows, csv_path)


def generate_diagnostic_plots(
    image_path: Path,
    preprocessed: np.ndarray,
    hog_cfg: HogConfig,
    lbp_cfg: LbpConfig,
    output_path: Path,
) -> None:
    original = plt.imread(image_path)
    _, hog_image = hog(
        preprocessed,
        orientations=hog_cfg.orientations,
        pixels_per_cell=hog_cfg.pixels_per_cell,
        cells_per_block=hog_cfg.cells_per_block,
        block_norm=hog_cfg.block_norm,
        transform_sqrt=hog_cfg.transform_sqrt,
        visualize=True,
        feature_vector=True,
    )
    lbp_image = local_binary_pattern(preprocessed, lbp_cfg.n_points, lbp_cfg.radius, method=lbp_cfg.method)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(preprocessed, cmap="gray")
    axes[0, 1].set_title("Preprocessed")
    axes[1, 0].imshow(hog_image, cmap="inferno")
    axes[1, 0].set_title("HOG")
    axes[1, 1].imshow(lbp_image, cmap="gray")
    axes[1, 1].set_title("LBP")

    for ax in axes.ravel():
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_optimization(
    csv_path: Path,
    output_dir: Path,
    config: OptimizationConfig,
    storage_url: str | None = None,
) -> optuna.study.Study:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    logs_dir = output_dir / "logs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "run.log")
    study: optuna.study.Study | None = None

    try:
        rows = load_dataset(csv_path)
        resolved = resolve_paths(rows)
        preprocessor = Preprocessor(PreprocessConfig())
        images, labels, kept = preprocess_images(resolved, preprocessor, logger)

        sample_image = images[0]
        validate_hog_params(sample_image.shape, 4, 1)

        log_rows: List[dict] = []
        study_db = output_dir / "study.db"
        storage = storage_url or f"sqlite:///{study_db}"

        sampler = TPESampler(seed=config.seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        study = optuna.create_study(
            study_name=config.study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        def objective(trial: optuna.trial.Trial) -> float:
            hog_cfg, lbp_cfg = sample_params(trial)
            try:
                validate_hog_params(sample_image.shape, hog_cfg.pixels_per_cell[0], hog_cfg.cells_per_block[0])
                validate_lbp_params(lbp_cfg.radius, lbp_cfg.n_points, lbp_cfg.method)
            except ValueError as exc:
                logger.warning("Trial %s invalid params: %s", trial.number, exc)
                raise optuna.exceptions.TrialPruned() from exc

            try:
                score, report = evaluate_trial(images, labels, hog_cfg, lbp_cfg, config.seed)
            except ValueError as exc:
                logger.warning("Trial %s failed: %s", trial.number, exc)
                raise optuna.exceptions.TrialPruned() from exc

            report_path = logs_dir / f"trial_{trial.number}_report.txt"
            report_path.write_text(report, encoding="utf-8")
            return score

        def log_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            log_trial(log_rows, trial)

        study.optimize(objective, n_trials=config.n_trials, callbacks=[log_callback])

        save_trial_log(log_rows, logs_dir / "trials.csv")
        save_top_k(log_rows, config.top_k, logs_dir / "top_k.csv")

        best_params = study.best_trial.params
        best_params_path = output_dir / "best_params.json"
        best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

        best_hog = HogConfig(
            orientations=best_params["hog_orientations"],
            pixels_per_cell=(best_params["hog_pixels_per_cell"], best_params["hog_pixels_per_cell"]),
            cells_per_block=(best_params["hog_cells_per_block"], best_params["hog_cells_per_block"]),
        )
        best_lbp = LbpConfig(
            radius=best_params["lbp_radius"],
            n_points=8 * best_params["lbp_radius"],
            method="uniform",
        )

        plot_path = plots_dir / "best_trial.png"
        generate_diagnostic_plots(kept[0][0], images[0], best_hog, best_lbp, plot_path)

        logger.info("Best score: %.4f", study.best_value)
        logger.info("Best params: %s", best_params)

        return study
    finally:
        if study is not None:
            storage = getattr(study, "_storage", None)
            if storage is not None:
                if hasattr(storage, "remove_session"):
                    storage.remove_session()
                if hasattr(storage, "engine"):
                    storage.engine.dispose()
                if hasattr(storage, "dispose"):
                    storage.dispose()
        close_logger(logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize HOG+LBP features with Optuna")
    parser.add_argument("--dataset-csv", default=DEFAULT_DATASET_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OptimizationConfig(
        n_trials=args.n_trials,
        top_k=args.top_k,
        seed=args.seed,
    )

    run_optimization(
        csv_path=Path(args.dataset_csv),
        output_dir=Path(args.output_dir),
        config=config,
    )


if __name__ == "__main__":
    main()
