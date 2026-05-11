import unittest
from pathlib import Path

from scripts.filter_dataset import (
    DEFAULT_SECTION,
    DEFAULT_SUBSET,
    DatasetConfig,
    filter_rows,
    normalize_filepath,
    parse_label,
    parse_required_count,
    parse_subset,
)


def _row(section: str, label: str, participant: str = "P1", filename: str = "img.jpg"):
    return {
        "Signo": label,
        "Path": f"{section}/{participant}/{label}/{filename}",
        "Participante": participant,
    }


class TestFilterLsc70wDataset(unittest.TestCase):
    def test_parse_label_trims_whitespace(self) -> None:
        row = {"Signo": "  HOLA  ", "Path": "LSC70W/Per01/HOLA/img.jpg"}
        self.assertEqual(parse_label(row), "HOLA")

    def test_default_config_targets_lsc70anh_abcde(self) -> None:
        config = DatasetConfig(required_count=None)

        self.assertEqual(config.section, DEFAULT_SECTION)
        self.assertEqual(config.subset, DEFAULT_SUBSET)

    def test_filter_rows_uses_section_and_subset(self) -> None:
        rows = [
            _row("LSC70ANH", "A", "P1", "2.jpg"),
            _row("LSC70ANH", "B", "P2", "1.jpg"),
            _row("LSC70W", "A", "P3", "3.jpg"),
            _row("LSC70ANH", "C", "P4", "4.jpg"),
        ]
        filtered = filter_rows(
            rows,
            config=DatasetConfig(subset=("A", "B"), required_count=None),
        )

        self.assertEqual([row["label"] for row in filtered], ["A", "B"])
        self.assertTrue(all("LSC70ANH" in row["filepath"] for row in filtered))

    def test_filter_rows_requires_exact_counts(self) -> None:
        allowlist = ("A", "B")
        rows = [
            _row("LSC70ANH", "A", "P1", "1.jpg"),
            _row("LSC70ANH", "A", "P2", "2.jpg"),
            _row("LSC70ANH", "B", "P3", "3.jpg"),
        ]
        with self.assertRaises(ValueError):
            filter_rows(rows, allowlist=allowlist, required_count=2)

    def test_filter_rows_skips_missing_participant(self) -> None:
        rows = [_row("LSC70ANH", "HOLA", participant="")]
        with self.assertRaisesRegex(ValueError, "empty"):
            filter_rows(rows, allowlist=("HOLA",), required_count=0)

    def test_filter_rows_normalizes_filepath(self) -> None:
        self.assertEqual(
            normalize_filepath("LSC70W/Per01/HOLA/img.jpg", Path("data/raw")),
            "data/raw/LSC70W/Per01/HOLA/img.jpg",
        )

    def test_invalid_section_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Section not found"):
            filter_rows(
                [_row("LSC70W", "A")],
                config=DatasetConfig(section="LSC70ANH", required_count=None),
            )

    def test_missing_labels_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "Requested labels not found"):
            filter_rows(
                [_row("LSC70ANH", "A")],
                config=DatasetConfig(subset=("A", "B"), required_count=None),
            )

    def test_empty_filtered_dataset_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            filter_rows(
                [_row("LSC70ANH", "A", participant="")],
                config=DatasetConfig(subset=("A",), required_count=None),
            )

    def test_cli_value_parsers(self) -> None:
        self.assertEqual(parse_subset("A, B,C"), ("A", "B", "C"))
        self.assertIsNone(parse_required_count("none"))
        self.assertEqual(parse_required_count("5"), 5)


if __name__ == "__main__":
    unittest.main()
