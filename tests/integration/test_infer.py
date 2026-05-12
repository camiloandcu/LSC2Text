from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ml.feature_extraction import FeatureConfig, extract_features
from src.ml.preprocessing import Preprocessor


def _write_image(path: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (255, 255, 255)
    cv2.imwrite(str(path), image)


class TestInferIntegration(unittest.TestCase):
    def test_infer_cli_returns_json_predictions(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            model_path = root / "model.joblib"
            _write_image(image_path)

            preprocessor = Preprocessor()
            feature_config = FeatureConfig()
            sample_image = preprocessor.process_path(str(image_path))
            sample_vector = extract_features(sample_image, feature_config)
            feature_length = int(sample_vector.shape[0])

            rng = np.random.default_rng(123)
            features = rng.normal(size=(24, feature_length)).astype(np.float32)
            labels = np.array(["A"] * 12 + ["B"] * 12)
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="linear", probability=True, random_state=0)),
                ]
            )
            model.fit(features, labels)
            joblib.dump(model, model_path)

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "-m",
                    "scripts.infer",
                    "--image",
                    str(image_path),
                    "--model-path",
                    str(model_path),
                    "--top-k",
                    "2",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            payload = json.loads(result.stdout)
            self.assertIn("timestamp", payload)
            self.assertEqual(payload["model"], str(model_path))
            self.assertIn("predictions", payload)
            self.assertEqual(len(payload["predictions"]), 2)
            self.assertEqual(payload["predictions"][0]["label"], "B")
            self.assertGreater(payload["predictions"][0]["confidence"], payload["predictions"][1]["confidence"])


if __name__ == "__main__":
    unittest.main()