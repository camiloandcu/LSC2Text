from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_model(model: Any, features: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    preds = model.predict(features)
    f1_macro = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, zero_division=0)
    matrix = confusion_matrix(labels, preds)

    return {
        "f1_macro": float(f1_macro),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
