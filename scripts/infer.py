from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.ml.feature_extraction import FeatureConfig, extract_features
from src.ml.preprocessing import Preprocessor


DEFAULT_MODEL_PATH = "models/registry/svm/direct-svm-20260511-220859/model.joblib"
DEFAULT_TOP_K = 3
DEFAULT_TIMEOUT = 5


def softmax(logits: np.ndarray) -> np.ndarray:
    ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def json_error(message: str, details: dict | None = None) -> str:
    payload = {"error": message}
    if details:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image inference using a trained model")

    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Path to model.joblib (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Top-K predictions to return (default: {DEFAULT_TOP_K})")
    parser.add_argument("--calibrate", action="store_true", help="Apply calibration if a calibrator is available alongside the model")
    parser.add_argument("--output", help="Path to write JSON output (default: stdout)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Max seconds allowed (default: {DEFAULT_TIMEOUT})")

    return parser.parse_args()


def run_inference(image_path: Path, model_path: Path, top_k: int = 3, calibrate: bool = False) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    # Load model
    model = joblib.load(model_path)

    # Preprocess and extract features
    preprocessor = Preprocessor()
    feature_config = FeatureConfig()

    image = preprocessor.process_path(str(image_path))
    vector = extract_features(image, feature_config)
    X = np.asarray(vector, dtype=np.float32).reshape(1, -1)

    # Optionally apply calibrator
    calibrator_path = model_path.parent / "calibrator.joblib"
    calibrator = None
    if calibrate and calibrator_path.exists():
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception:
            calibrator = None

    # Obtain probabilities
    probs: np.ndarray
    classes: list[Any]
    if calibrator is not None and hasattr(calibrator, "predict_proba"):
        probs = calibrator.predict_proba(X)
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
        else:
            classes = [str(i) for i in range(probs.shape[1])]
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = list(model.classes_) if hasattr(model, "classes_") else [str(i) for i in range(probs.shape[1])]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            logits = np.vstack((-scores, scores)).T
        else:
            logits = scores
        probs = softmax(np.asarray(logits, dtype=float))
        classes = list(model.classes_) if hasattr(model, "classes_") else [str(i) for i in range(probs.shape[1])]
    else:
        # Last resort: model.predict and produce one-hot probability for predicted class
        pred = model.predict(X)
        classes = list(model.classes_) if hasattr(model, "classes_") else [str(pred[0])]
        probs = np.zeros((1, len(classes)), dtype=float)
        idx = classes.index(pred[0]) if pred[0] in classes else 0
        probs[0, idx] = 1.0

    # Select top-k
    top_k = max(1, min(top_k, probs.shape[1]))
    indices = np.argsort(probs[0])[::-1][:top_k]
    predictions = [
        {"label": classes[int(i)], "confidence": float(probs[0, int(i)])}
        for i in indices
    ]

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": str(model_path),
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("infer")

    image_path = Path(args.image)
    model_path = Path(args.model_path)

    try:
        result = run_inference(image_path, model_path, top_k=args.top_k, calibrate=args.calibrate)
        output_json = json.dumps(result, ensure_ascii=False)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(output_json)
        else:
            print(output_json)

        sys.exit(0)

    except FileNotFoundError as fe:
        logger.error("%s", str(fe))
        print(json.dumps({"error": str(fe)}), file=sys.stdout)
        sys.exit(3)
    except Exception as exc:
        logger.exception("Inference failed")
        print(json.dumps({"error": "inference failed", "details": str(exc)}), file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
