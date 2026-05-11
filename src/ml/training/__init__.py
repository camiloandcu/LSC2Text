"""Training and evaluation utilities for classical models."""

from .train import train_mlp, train_svm, persist_artifacts
from .evaluate import evaluate_model

from .optimize import (
    OptimizationConfig,
    create_median_pruner,
    create_objective,
    params_from_trial,
    run_optimization,
    sample_mlp_params,
    sample_svm_params,
    validate_params,
)

__all__ = [
    "train_mlp", 
    "train_svm", 
    "persist_artifacts", 
    "evaluate_model",
    "OptimizationConfig",
    "create_median_pruner",
    "create_objective",
    "params_from_trial",
    "run_optimization",
    "sample_mlp_params",
    "sample_svm_params",
    "validate_params",
]
