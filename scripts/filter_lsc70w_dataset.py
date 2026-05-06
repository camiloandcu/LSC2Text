# ///script
# dependencies =[
# ]
# ///

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_ALLOWLIST = (
    "AÑOS",
    "BUENAS",
    "DIAS",
    "GUSTAR",
    "HOLA",
    "LICOR",
    "NOCHES",
    "NOMBRE",
    "TARDES",
    "YO",
)
REQUIRED_COLUMNS = ("Signo", "Path", "Participante")
DEFAULT_DATASET_ROOT = Path("data/raw/LSC70")


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            missing_labels = ", ".join(missing)
            raise ValueError(f"Missing required columns: {missing_labels}")

        return list(reader)


def parse_label(row: dict[str, str]) -> str:
    label = (row.get("Signo") or "").strip()
    if not label:
        raise ValueError("Missing label in Signo column")
    return label


def parse_path(row: dict[str, str]) -> str:
    path_value = (row.get("Path") or "").strip()
    if not path_value:
        raise ValueError("Missing path in Path column")
    return path_value


def parse_participant(row: dict[str, str]) -> str:
    return (row.get("Participante") or "").strip()


def normalize_filepath(path_value: str, dataset_root: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        raise ValueError("Path must be relative to dataset root")
    if ".." in path.parts:
        raise ValueError("Path cannot contain parent traversal segments")

    root_parts = dataset_root.parts
    if path.parts[: len(root_parts)] == root_parts:
        normalized = path
    else:
        normalized = dataset_root / path
    return normalized.as_posix()


def filter_rows(
    rows: Iterable[dict[str, str]],
    allowlist: Iterable[str] = DEFAULT_ALLOWLIST,
    required_count: int = 420,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> list[dict[str, str]]:
    allowset = set(allowlist)
    filtered: list[dict[str, str]] = []

    for row in rows:
        label = parse_label(row)
        path_value = parse_path(row)
        participant = parse_participant(row)
        if not participant:
            continue
        filepath = normalize_filepath(path_value, dataset_root)
        if label not in allowset:
            continue
        filtered.append(
            {"filepath": filepath, "label": label, "participant": participant}
        )

    filtered.sort(key=lambda item: (item["label"], item["filepath"]))

    counts = Counter(item["label"] for item in filtered)
    mismatched = [
        label for label in allowlist if counts.get(label, 0) != required_count
    ]
    if mismatched:
        details = ", ".join(
            f"{label}={counts.get(label, 0)}" for label in mismatched
        )
        raise ValueError(
            "Class counts must match the required total. "
            f"Expected {required_count} each. Found: {details}"
        )

    return filtered


def write_filtered_csv(rows: Iterable[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filepath", "label", "participant"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter LSC70W samples from an existing dataset CSV."
    )
    parser.add_argument(
        "--input",
        default=Path("data/raw/dataset.csv"),
        type=Path,
        help="Input CSV path (default: data/raw/dataset.csv).",
    )
    parser.add_argument(
        "--output",
        default=Path("data/interim/dataset_lsc70w.csv"),
        type=Path,
        help="Output CSV path (default: data/interim/dataset_lsc70w.csv).",
    )
    args = parser.parse_args()

    try:
        rows = load_csv_rows(args.input)
        filtered = filter_rows(rows)
        write_filtered_csv(filtered, args.output)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {len(filtered)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
