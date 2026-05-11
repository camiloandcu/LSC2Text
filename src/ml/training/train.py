from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .evaluate import evaluate_model


DEFAULT_SEED = 1337


def _log_convergence_warnings(caught: list[warnings.WarningMessage], logger: logging.Logger) -> None:
    for warning in caught:
        if isinstance(warning.message, ConvergenceWarning):
            logger.warning("%s", warning.message)


def train_svm(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = DEFAULT_SEED,
    prefer_gpu: bool = True,
    **kwargs: Any,
) -> Pipeline:
    logger = logging.getLogger("model_training")
    if prefer_gpu:
        logger.info("GPU backend not available; using CPU (scikit-learn)")

    model_params = {"verbose": True, "max_iter": 1000, "random_state": seed, **kwargs}
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(**model_params)),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, labels)
        _log_convergence_warnings(caught, logger)
    return model


def train_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = DEFAULT_SEED,
    prefer_gpu: bool = True,
    **kwargs: Any,
) -> MLPClassifier:
    logger = logging.getLogger("model_training")
    if prefer_gpu:
        logger.info("GPU backend not available; using CPU (scikit-learn)")

    model_params = {"random_state": seed, "max_iter": 1000, **kwargs}
    model = MLPClassifier(**model_params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, labels)
        _log_convergence_warnings(caught, logger)
    return model


def persist_artifacts(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    registry_dir: Path,
    model_name: str,
    run_id: str | None = None,
) -> Dict[str, Path]:
    run_id = run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_dir = registry_dir / model_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_model(model, features, labels)

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {"model": model_path, "metrics": metrics_path, "run_dir": output_dir}
