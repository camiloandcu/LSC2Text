import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.optimize_features import (
    OptimizationConfig,
    run_optimization,
    validate_hog_params,
    validate_lbp_params,
)
from src.ml.features.feature_extraction import FeatureConfig, extract_features


def _write_image(path: Path) -> None:
    import cv2

    image = (np.random.rand(32, 32) * 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def _write_dataset(csv_path: Path, root: Path, count: int = 4) -> None:
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "label"])
        writer.writeheader()
        for idx in range(count):
            filename = f"img_{idx}.png"
            _write_image(root / filename)
            writer.writerow({"filepath": root / filename, "label": f"label_{idx % 2}"})


class TestFeatureOptimization(unittest.TestCase):
    def test_validate_params(self):
        validate_hog_params((32, 32), pixels_per_cell=8, cells_per_block=2)
        with self.assertRaises(ValueError):
            validate_hog_params((30, 30), pixels_per_cell=16, cells_per_block=2)

        validate_lbp_params(radius=2, n_points=16, method="uniform")
        with self.assertRaises(ValueError):
            validate_lbp_params(radius=2, n_points=8, method="uniform")

    def test_single_trial_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_csv = root / "dataset.csv"
            _write_dataset(dataset_csv, root, count=6)

            output_dir = root / "artifacts"
            config = OptimizationConfig(n_trials=1, top_k=1, seed=123)

            study = run_optimization(
                csv_path=dataset_csv,
                output_dir=output_dir,
                config=config,
                storage_url="sqlite:///:memory:",
            )

            self.assertTrue((output_dir / "best_params.json").exists())
            self.assertTrue((output_dir / "logs" / "trials.csv").exists())
            self.assertEqual(study.best_trial.number, 0)

    def test_feature_vector_consistency(self):
        image_a = (np.random.rand(64, 64) * 255).astype(np.float32)
        image_b = (np.random.rand(64, 64) * 255).astype(np.float32)
        config = FeatureConfig()

        vec_a = extract_features(image_a, config)
        vec_b = extract_features(image_b, config)

        self.assertEqual(vec_a.shape, vec_b.shape)
        self.assertEqual(len(vec_a.shape), 1)

    def test_smoke_small_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_csv = root / "dataset.csv"
            _write_dataset(dataset_csv, root, count=6)

            output_dir = root / "artifacts"
            config = OptimizationConfig(n_trials=2, top_k=1, seed=123)

            study = run_optimization(
                csv_path=dataset_csv,
                output_dir=output_dir,
                config=config,
                storage_url="sqlite:///:memory:",
            )

            self.assertGreaterEqual(len(study.trials), 2)


if __name__ == "__main__":
    unittest.main()
