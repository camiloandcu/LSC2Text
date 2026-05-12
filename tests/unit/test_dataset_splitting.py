import csv
import tempfile
import unittest
from pathlib import Path

from scripts.split_dataset import DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_CSV, compute_splits, load_rows, validate_split


def _write_dataset(csv_path: Path, participants: int = 6, per_participant: int = 2) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "label", "participant"])
        writer.writeheader()
        for p in range(participants):
            label = "A" if p % 2 == 0 else "B"
            participant = f"P{p:02d}"
            for i in range(per_participant):
                writer.writerow({
                    "filepath": f"img_{participant}_{i}.png",
                    "label": label,
                    "participant": participant,
                })


def _write_imbalanced(csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "label", "participant"])
        writer.writeheader()
        writer.writerow({"filepath": "img1.png", "label": "A", "participant": "P1"})
        writer.writerow({"filepath": "img2.png", "label": "B", "participant": "P2"})


class TestDatasetSplitting(unittest.TestCase):
    def test_defaults_are_generic_configured_dataset_paths(self):
        self.assertEqual(DEFAULT_INPUT_CSV, "data/processed/dataset_lsc70anh_abcde.csv")
        self.assertEqual(DEFAULT_OUTPUT_CSV, "data/splits/dataset_lsc70anh_abcde.csv")

    def test_balance_and_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "dataset.csv"
            _write_dataset(csv_path, participants=10, per_participant=2)
            rows = load_rows(csv_path)

            train_idx, val_idx, _ = compute_splits(rows, val_ratio=0.2, seed=123)
            validate_split(rows, train_idx, val_idx)

    def test_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "dataset.csv"
            _write_dataset(csv_path, participants=10, per_participant=2)
            rows = load_rows(csv_path)

            train_a, val_a, _ = compute_splits(rows, val_ratio=0.2, seed=123)
            train_b, val_b, _ = compute_splits(rows, val_ratio=0.2, seed=123)

            self.assertEqual(train_a, train_b)
            self.assertEqual(val_a, val_b)

    def test_imbalance_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "dataset.csv"
            _write_imbalanced(csv_path)
            rows = load_rows(csv_path)

            with self.assertRaises(ValueError):
                compute_splits(rows, val_ratio=0.5, seed=123)


if __name__ == "__main__":
    unittest.main()
