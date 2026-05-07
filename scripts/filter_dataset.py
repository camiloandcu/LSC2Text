# /// script
# dependencies =[
# ]
# ///

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SECTION = "LSC70ANH"
DEFAULT_SUBSET = ("A", "B", "C", "D", "E")
REQUIRED_COLUMNS = ("Signo", "Path", "Participante")
DEFAULT_DATASET_ROOT = Path("data/raw/LSC70")
DEFAULT_INPUT_CSV = Path("data/raw/dataset.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/dataset_lsc70anh_abcde.csv")
DEFAULT_REQUIRED_COUNT = 420


@dataclass(frozen=True)
class DatasetConfig:
    section: str = DEFAULT_SECTION
    subset: tuple[str, ...] = DEFAULT_SUBSET
    dataset_root: Path = DEFAULT_DATASET_ROOT
    input_path: Path = DEFAULT_INPUT_CSV
    output_path: Path = DEFAULT_OUTPUT_CSV
    required_count: int | None = DEFAULT_REQUIRED_COUNT


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


def parse_section(path_value: str) -> str:
    path = Path(path_value)
    if not path.parts:
        raise ValueError("Missing section in Path column")
    return path.parts[0]


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
    allowlist: Iterable[str] | None = None,
    required_count: int | None = DEFAULT_REQUIRED_COUNT,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    section: str | None = None,
    config: DatasetConfig | None = None,
) -> list[dict[str, str]]:
    if config is not None:
        allowlist = config.subset
        required_count = config.required_count
        dataset_root = config.dataset_root
        section = config.section

    selected_labels = tuple(allowlist or DEFAULT_SUBSET)
    allowset = set(selected_labels)
    filtered: list[dict[str, str]] = []
    available_sections: set[str] = set()
    labels_in_section: set[str] = set()

    for row in rows:
        label = parse_label(row)
        path_value = parse_path(row)
        row_section = parse_section(path_value)
        available_sections.add(row_section)
        if section is not None and row_section != section:
            continue

        labels_in_section.add(label)
        participant = parse_participant(row)
        if not participant:
            continue
        if label not in allowset:
            continue

        filepath = normalize_filepath(path_value, dataset_root)
        filtered.append({"filepath": filepath, "label": label, "participant": participant})

    if section is not None and section not in available_sections:
        available = ", ".join(sorted(available_sections)) or "<none>"
        raise ValueError(f"Section not found: {section}. Available sections: {available}")

    missing_labels = sorted(allowset.difference(labels_in_section))
    if missing_labels:
        missing = ", ".join(missing_labels)
        available = ", ".join(sorted(labels_in_section)) or "<none>"
        raise ValueError(
            f"Requested labels not found in section {section or '<all>'}: {missing}. "
            f"Available labels: {available}"
        )

    if not filtered:
        raise ValueError("Filtered dataset is empty")

    filtered.sort(key=lambda item: (item["label"], item["filepath"]))

    counts = Counter(item["label"] for item in filtered)
    if required_count is not None:
        mismatched = [
            label for label in selected_labels if counts.get(label, 0) != required_count
        ]
    else:
        mismatched = []
    if required_count is not None and mismatched:
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
        writer = csv.DictWriter(handle, fieldnames=["filepath", "label", "participant"])
        writer.writeheader()
        writer.writerows(rows)


def parse_subset(value: str) -> tuple[str, ...]:
    labels = tuple(label.strip() for label in value.split(",") if label.strip())
    if not labels:
        raise argparse.ArgumentTypeError("subset must include at least one label")
    return labels


def parse_required_count(value: str) -> int | None:
    if value.lower() in {"none", "off", "false", "no"}:
        return None
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("required count must be non-negative or 'none'")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter LSC70 samples by configurable section and sign subset."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_CSV,
        type=Path,
        help=f"Input CSV path (default: {DEFAULT_INPUT_CSV}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        type=Path,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--section",
        default=DEFAULT_SECTION,
        help=f"Dataset section to include (default: {DEFAULT_SECTION}).",
    )
    parser.add_argument(
        "--subset",
        default=",".join(DEFAULT_SUBSET),
        type=parse_subset,
        help="Comma-separated labels to include (default: A,B,C,D,E).",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        type=Path,
        help=f"Dataset root prepended to manifest paths (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--required-count",
        default=str(DEFAULT_REQUIRED_COUNT),
        type=parse_required_count,
        help="Required image count per selected label, or 'none' to disable.",
    )
    args = parser.parse_args()

    try:
        config = DatasetConfig(
            section=args.section,
            subset=args.subset,
            dataset_root=args.dataset_root,
            input_path=args.input,
            output_path=args.output,
            required_count=args.required_count,
        )
        rows = load_csv_rows(config.input_path)
        filtered = filter_rows(rows, config=config)
        write_filtered_csv(filtered, config.output_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "Dataset config: "
        f"section={config.section} subset={','.join(config.subset)} "
        f"input={config.input_path} output={config.output_path}"
    )
    print(f"Wrote {len(filtered)} rows to {config.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
