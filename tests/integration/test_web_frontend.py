from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.api.api import ApiSettings, create_app
from src.ml.feature_extraction import FeatureConfig, extract_features
from src.ml.preprocessing import Preprocessor


def _write_image(path: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (255, 255, 255)
    cv2.imwrite(str(path), image)


class TestWebFrontend(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "sample.png"
        self.model_path = self.root / "model.joblib"
        _write_image(self.image_path)

        preprocessor = Preprocessor()
        feature_config = FeatureConfig()
        sample_image = preprocessor.process_path(str(self.image_path))
        feature_length = int(extract_features(sample_image, feature_config).shape[0])

        rng = np.random.default_rng(123)
        features = rng.normal(size=(27, feature_length)).astype(np.float32)
        labels = np.array(["A"] * 9 + ["B"] * 9 + ["C"] * 9)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="linear", probability=True, random_state=0)),
            ]
        )
        model.fit(features, labels)
        joblib.dump(model, self.model_path)

        settings = ApiSettings(model_path=self.model_path, top_k=3)
        self.client = TestClient(create_app(settings))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_render_upload_page(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Upload and Predict", response.text)
        self.assertIn("/frontend/predict", response.text)

    def test_successful_upload_renders_result(self) -> None:
        with self.image_path.open("rb") as handle:
            response = self.client.post(
                "/frontend/predict",
                files={"image": (self.image_path.name, handle, "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Prediction Result", response.text)
        self.assertIn("Predict Another Image", response.text)
        self.assertGreaterEqual(response.text.count("Ranked prediction"), 3)
        self.assertIn("%", response.text)

    def test_invalid_file_renders_error(self) -> None:
        response = self.client.post(
            "/frontend/predict",
            files={"image": ("bad.txt", b"not-an-image", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unable to Process Your Upload", response.text)
        self.assertIn("not a valid image", response.text)


if __name__ == "__main__":
    unittest.main()
