import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.optimize_train import load_split_data, resolve_manifest_path


def _write_image(path: Path) -> None:
    image = (np.random.rand(32, 32) * 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def _write_manifest(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "label", "participant", "split"])
        writer.writeheader()
        writer.writerows(rows)


class TestOptimizeTrainDataLoading(unittest.TestCase):
    def test_manifest_loads_images_into_feature_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_a = root / "train_a.png"
            train_b = root / "train_b.png"
            val_a = root / "val_a.png"
            ignored = root / "ignored.png"
            for path in [train_a, train_b, val_a, ignored]:
                _write_image(path)

            csv_path = root / "split.csv"
            _write_manifest(
                csv_path,
                [
                    {"filepath": "train_a.png", "label": "A", "participant": "P1", "split": "train"},
                    {"filepath": str(train_b), "label": "B", "participant": "P2", "split": "train"},
                    {"filepath": "val_a.png", "label": "A", "participant": "P3", "split": "val"},
                    {"filepath": "ignored.png", "label": "B", "participant": "P4", "split": "test"},
                ],
            )

            train_features, train_labels, val_features, val_labels = load_split_data(csv_path)

            self.assertEqual(train_features.shape[0], 2)
            self.assertEqual(val_features.shape[0], 1)
            self.assertEqual(train_features.shape[1], val_features.shape[1])
            self.assertEqual(train_features.dtype, np.float32)
            self.assertEqual(train_labels.tolist(), ["A", "B"])
            self.assertEqual(val_labels.tolist(), ["A"])

    def test_resolve_manifest_path_prefers_existing_cwd_path_then_csv_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "split.csv"

            resolved = resolve_manifest_path("image.png", csv_path)

            self.assertEqual(resolved, csv_path.parent / "image.png")

    def test_missing_required_manifest_columns_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "split.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["label", "split"])
                writer.writeheader()
                writer.writerow({"label": "A", "split": "train"})

            with self.assertRaisesRegex(ValueError, "filepath, label, split"):
                load_split_data(csv_path)

    def test_missing_validation_partition_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "train.png"
            _write_image(image_path)
            csv_path = root / "split.csv"
            _write_manifest(
                csv_path,
                [{"filepath": "train.png", "label": "A", "participant": "P1", "split": "train"}],
            )

            with self.assertRaisesRegex(ValueError, "no validation rows"):
                load_split_data(csv_path)

    def test_unreadable_image_error_includes_row_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "split.csv"
            _write_manifest(
                csv_path,
                [{"filepath": "missing.png", "label": "A", "participant": "P1", "split": "train"}],
            )

            with self.assertRaisesRegex(OSError, "row 2.*missing.png"):
                load_split_data(csv_path)


if __name__ == "__main__":
    unittest.main()
