"""Training and evaluation utilities for classical models."""

from .train import train_mlp, train_svm, persist_artifacts
from .evaluate import evaluate_model

__all__ = [
    "train_mlp", 
    "train_svm", 
    "persist_artifacts", 
    "evaluate_model",
]
