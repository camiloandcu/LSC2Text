from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator
import unittest

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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"server did not become ready: {last_error}")


def _multipart_body(field_name: str, filename: str, content_type: str, payload: bytes) -> tuple[bytes, str]:
    boundary = "----LSC2TextBoundary7b3b6b8d"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            payload,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


class TestApiIntegration(unittest.TestCase):
    def test_health_metadata_and_predict_endpoints(self) -> None:
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
            features = rng.normal(size=(27, feature_length)).astype(np.float32)
            labels = np.array(["A"] * 9 + ["B"] * 9 + ["C"] * 9)
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="linear", probability=True, random_state=0)),
                ]
            )
            model.fit(features, labels)
            joblib.dump(model, model_path)

            port = _free_port()
            process = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "-m",
                    "src.api.api",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--model-path",
                    str(model_path),
                ],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                base_url = f"http://127.0.0.1:{port}"
                _wait_for_health(f"{base_url}/health")

                with urllib.request.urlopen(f"{base_url}/health") as response:
                    self.assertEqual(response.status, 200)
                    payload = json.loads(response.read())
                    self.assertTrue(payload["ready"])

                with urllib.request.urlopen(f"{base_url}/metadata") as response:
                    self.assertEqual(response.status, 200)
                    payload = json.loads(response.read())
                    self.assertIn("/predict", " ".join(payload["endpoints"]))
                    self.assertEqual(payload["default_model_path"], str(model_path))

                image_bytes = image_path.read_bytes()
                body, content_type = _multipart_body("image", image_path.name, "image/png", image_bytes)
                request = urllib.request.Request(
                    f"{base_url}/predict",
                    data=body,
                    method="POST",
                    headers={"Content-Type": content_type},
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 200)
                    payload = json.loads(response.read())
                    self.assertIn("timestamp", payload)
                    self.assertEqual(payload["model"], str(model_path))
                    self.assertEqual(len(payload["predictions"]), 3)

                invalid_body, invalid_content_type = _multipart_body("image", "bad.txt", "text/plain", b"not-an-image")
                invalid_request = urllib.request.Request(
                    f"{base_url}/predict",
                    data=invalid_body,
                    method="POST",
                    headers={"Content-Type": invalid_content_type},
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(invalid_request)
                self.assertEqual(ctx.exception.code, 400)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()