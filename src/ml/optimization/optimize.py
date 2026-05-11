from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.ml.training import evaluate_model, persist_artifacts, train_mlp, train_svm


DEFAULT_SEED = 1337
SUPPORTED_MODELS = {"svm", "mlp"}
TrialParams = Dict[str, Any]
SearchSpace = Callable[[optuna.trial.Trial], TrialParams]


@dataclass
class OptimizationConfig:
    model_type: str
    n_trials: int = 30
    seed: int = DEFAULT_SEED
    study_name: str = "hyperparam_optimization"
    output_dir: Path = Path("artifacts/experiments/hyperparam_optimization")
    registry_dir: Path = Path("models/registry")
    pruner_startup_trials: int = 5
    pruner_warmup_steps: int = 0


def sample_svm_params(trial: optuna.trial.Trial) -> TrialParams:
    return {
        "C": trial.suggest_float("C", 0.1, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
    }


def sample_mlp_params(trial: optuna.trial.Trial) -> TrialParams:
    hidden_key = trial.suggest_categorical("hidden_layer_sizes", ["64", "128", "64_32"])
    hidden_sizes = {
        "64": (64,),
        "128": (128,),
        "64_32": (64, 32),
    }[hidden_key]
    return {
        "hidden_layer_sizes": hidden_sizes,
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


def params_from_trial(model_type: str, params: TrialParams) -> TrialParams:
    _validate_model_type(model_type)
    if model_type == "svm":
        train_params = {key: params[key] for key in ("C", "gamma")}
    else:
        train_params = dict(params)
        if isinstance(train_params.get("hidden_layer_sizes"), str):
            train_params["hidden_layer_sizes"] = {
                "64": (64,),
                "128": (128,),
                "64_32": (64, 32),
            }[train_params["hidden_layer_sizes"]]
    validate_params(model_type, train_params)
    return train_params


def validate_params(model_type: str, params: TrialParams) -> None:
    _validate_model_type(model_type)
    if model_type == "svm":
        if params["C"] <= 0 or params["gamma"] <= 0:
            raise ValueError("SVM C and gamma must be positive")
        return

    hidden = params["hidden_layer_sizes"]
    if not isinstance(hidden, tuple) or not hidden or any(size <= 0 for size in hidden):
        raise ValueError("MLP hidden_layer_sizes must be a tuple of positive layer sizes")
    if params["alpha"] <= 0 or params["learning_rate_init"] <= 0:
        raise ValueError("MLP alpha and learning_rate_init must be positive")


def create_median_pruner(config: OptimizationConfig) -> MedianPruner:
    return MedianPruner(
        n_startup_trials=config.pruner_startup_trials,
        n_warmup_steps=config.pruner_warmup_steps,
    )


def create_objective(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: OptimizationConfig,
    search_space: SearchSpace | None = None,
) -> Callable[[optuna.trial.Trial], float]:
    _validate_model_type(config.model_type)
    sampler = search_space or _search_space_for(config.model_type)
    logger = logging.getLogger("hyperparam_optimization")

    def objective(trial: optuna.trial.Trial) -> float:
        try:
            params = sampler(trial)
            validate_params(config.model_type, params)
            model = _train_model(config.model_type, train_features, train_labels, config.seed, params)
            metrics = evaluate_model(model, val_features, val_labels)
        except Exception as exc:
            logger.warning("Trial %s pruned: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned() from exc

        score = float(metrics["f1_macro"])
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return score

    return objective


def run_optimization(
    features: np.ndarray,
    labels: np.ndarray,
    config: OptimizationConfig,
    validation_features: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    storage_url: str | None = None,
    search_space: SearchSpace | None = None,
) -> Tuple[optuna.study.Study, Dict[str, Path]]:
    _validate_model_type(config.model_type)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.registry_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = config.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    val_features = features if validation_features is None else validation_features
    val_labels = labels if validation_labels is None else validation_labels

    storage_path = config.output_dir / "study.db"
    storage = storage_url or f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name=config.study_name,
        direction="maximize",
        sampler=TPESampler(seed=config.seed),
        pruner=create_median_pruner(config),
        storage=storage,
        load_if_exists=True,
    )

    log_rows: List[dict] = []

    def log_callback(_: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        row = {
            "trial": trial.number,
            "score": trial.value if trial.value is not None else "",
            "state": trial.state.name,
            **trial.params,
        }
        log_rows.append(row)
        _write_trial_log(log_rows, logs_dir / "trials.csv")

    objective = create_objective(features, labels, val_features, val_labels, config, search_space)
    study.optimize(objective, n_trials=config.n_trials, callbacks=[log_callback])

    best_params = params_from_trial(config.model_type, study.best_trial.params)
    params_path = config.output_dir / "best_params.json"
    payload = {
        "model_type": config.model_type,
        "score": study.best_value,
        "params": _jsonable_params(best_params),
        "trial_params": _jsonable_params(study.best_trial.params),
    }
    params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best_model = _train_model(config.model_type, features, labels, config.seed, best_params)
    artifacts = persist_artifacts(
        best_model,
        val_features,
        val_labels,
        config.registry_dir,
        config.model_type,
        run_id=f"{config.study_name}-{study.best_trial.number}",
    )

    return study, {"best_params": params_path, "trial_log": logs_dir / "trials.csv", **artifacts}


def _validate_model_type(model_type: str) -> None:
    if model_type not in SUPPORTED_MODELS:
        raise ValueError("model_type must be 'svm' or 'mlp'")


def _search_space_for(model_type: str) -> SearchSpace:
    if model_type == "svm":
        return sample_svm_params
    return sample_mlp_params


def _train_model(
    model_type: str,
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    params: TrialParams,
):
    if model_type == "svm":
        return train_svm(features, labels, seed=seed, prefer_gpu=False, **params)
    return train_mlp(features, labels, seed=seed, prefer_gpu=False, **params)


def _write_trial_log(rows: List[dict], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_params(params: TrialParams) -> TrialParams:
    return {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()}
