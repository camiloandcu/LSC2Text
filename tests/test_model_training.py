import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ml.training import evaluate_model, persist_artifacts, train_mlp, train_svm


class TestModelTraining(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.features = rng.normal(size=(40, 8)).astype(np.float32)
        self.labels = np.array(["A"] * 20 + ["B"] * 20)

    def test_training_runs(self):
        svm = train_svm(self.features, self.labels, prefer_gpu=False)
        mlp = train_mlp(self.features, self.labels, prefer_gpu=False)

        self.assertIsNotNone(svm)
        self.assertIsNotNone(mlp)

    def test_svm_uses_scaled_rbf_pipeline(self):
        svm = train_svm(self.features, self.labels, prefer_gpu=False, kernel="poly", C=2.0)

        self.assertIsInstance(svm, Pipeline)
        self.assertIsInstance(svm.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(svm.named_steps["svc"], SVC)
        self.assertEqual(svm.named_steps["svc"].kernel, "rbf")
        self.assertEqual(svm.named_steps["svc"].C, 2.0)

    def test_evaluation_outputs(self):
        svm = train_svm(self.features, self.labels, prefer_gpu=False)
        metrics = evaluate_model(svm, self.features, self.labels)

        self.assertIn("f1_macro", metrics)
        self.assertIn("classification_report", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_persistence_outputs(self):
        svm = train_svm(self.features, self.labels, prefer_gpu=False)
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry"
            outputs = persist_artifacts(svm, self.features, self.labels, registry, "svm", run_id="test")

            self.assertTrue(outputs["model"].exists())
            self.assertTrue(outputs["metrics"].exists())
            self.assertTrue(outputs["run_dir"].exists())


if __name__ == "__main__":
    unittest.main()
