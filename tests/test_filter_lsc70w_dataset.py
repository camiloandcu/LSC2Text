import unittest
from pathlib import Path

from scripts.filter_lsc70w_dataset import filter_rows, parse_label


class TestFilterLsc70wDataset(unittest.TestCase):
    def test_parse_label_trims_whitespace(self) -> None:
        row = {"Signo": "  HOLA  ", "Path": "LSC70W/Per01/HOLA/img.jpg"}
        self.assertEqual(parse_label(row), "HOLA")

    def test_filter_rows_requires_exact_counts(self) -> None:
        allowlist = ("A", "B")
        rows = [
            {"Signo": "A", "Path": "p1", "Participante": "P1"},
            {"Signo": "A", "Path": "p2", "Participante": "P2"},
            {"Signo": "B", "Path": "p3", "Participante": "P3"},
        ]
        with self.assertRaises(ValueError):
            filter_rows(rows, allowlist=allowlist, required_count=2)

    def test_filter_rows_skips_missing_participant(self) -> None:
        rows = [
            {"Signo": "HOLA", "Path": "LSC70W/Per01/HOLA/img.jpg", "Participante": ""}
        ]
        filtered = filter_rows(rows, allowlist=("HOLA",), required_count=0)
        self.assertEqual(filtered, [])

    def test_filter_rows_normalizes_filepath(self) -> None:
        rows = [
            {
                "Signo": "HOLA",
                "Path": "LSC70W/Per01/HOLA/img.jpg",
                "Participante": "Per01",
            }
        ]
        filtered = filter_rows(
            rows,
            allowlist=("HOLA",),
            required_count=1,
            dataset_root=Path("data/raw"),
        )
        self.assertEqual(
            filtered[0]["filepath"],
            "data/raw/LSC70W/Per01/HOLA/img.jpg",
        )


if __name__ == "__main__":
    unittest.main()
