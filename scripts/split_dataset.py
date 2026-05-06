from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_INPUT_CSV = "data/interim/dataset_lsc70w.csv"
DEFAULT_OUTPUT_CSV = "data/splits/dataset_lsc70w.csv"
DEFAULT_SEED = 1337
DEFAULT_VAL_RATIO = 0.2


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filepath", "label", "participant"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("dataset CSV must include filepath, label, participant columns")
        rows = [row for row in reader]
    if not rows:
        raise ValueError("dataset CSV contains no rows")
    return rows


def validate_class_counts(labels: List[str], n_splits: int) -> None:
    counts = Counter(labels)
    too_small = {label: count for label, count in counts.items() if count < n_splits}
    if too_small:
        detail = ", ".join(f"{label}={count}" for label, count in sorted(too_small.items()))
        raise ValueError(
            f"class imbalance prevents stratification with n_splits={n_splits}: {detail}"
        )


def compute_splits(
    rows: List[Dict[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], int]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    n_splits = max(2, int(round(1.0 / val_ratio)))

    labels = [row["label"] for row in rows]
    groups = [row["participant"] for row in rows]

    validate_class_counts(labels, n_splits)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(rows, labels, groups))
    return list(train_idx), list(val_idx), n_splits


def validate_split(rows: List[Dict[str, str]], train_idx: List[int], val_idx: List[int]) -> None:
    train_groups = {rows[i]["participant"] for i in train_idx}
    val_groups = {rows[i]["participant"] for i in val_idx}
    overlap = train_groups.intersection(val_groups)
    if overlap:
        raise ValueError(f"group leakage detected: {sorted(overlap)}")

    all_labels = {row["label"] for row in rows}
    train_labels = Counter(rows[i]["label"] for i in train_idx)
    val_labels = Counter(rows[i]["label"] for i in val_idx)

    missing_train = [label for label in all_labels if train_labels[label] == 0]
    missing_val = [label for label in all_labels if val_labels[label] == 0]
    if missing_train or missing_val:
        raise ValueError(
            f"class imbalance prevents stratification: missing train={missing_train} "
            f"missing val={missing_val}"
        )


def write_split_csv(
    rows: List[Dict[str, str]],
    train_idx: List[int],
    val_idx: List[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_set = set(train_idx)
    val_set = set(val_idx)

    fieldnames = list(rows[0].keys()) + ["split"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            if idx in train_set:
                split = "train"
            elif idx in val_set:
                split = "val"
            else:
                continue
            writer.writerow({**row, "split": split})


def summarize_split(rows: List[Dict[str, str]], train_idx: List[int], val_idx: List[int]) -> str:
    total = len(rows)
    train_labels = Counter(rows[i]["label"] for i in train_idx)
    val_labels = Counter(rows[i]["label"] for i in val_idx)
    labels = sorted({row["label"] for row in rows})

    lines = [f"Total rows: {total}", f"Train: {len(train_idx)}", f"Val: {len(val_idx)}"]
    lines.append("Class distribution:")
    for label in labels:
        lines.append(f"  {label}: train={train_labels[label]}, val={val_labels[label]}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified group splits for LSC70W")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input_csv))
    train_idx, val_idx, n_splits = compute_splits(rows, args.val_ratio, args.seed)
    validate_split(rows, train_idx, val_idx)
    write_split_csv(rows, train_idx, val_idx, Path(args.output_csv))
    summary = summarize_split(rows, train_idx, val_idx)
    print(summary)
    if abs(args.val_ratio - (1.0 / n_splits)) > 1e-6:
        print(f"Note: effective val ratio is {1.0 / n_splits:.3f} with n_splits={n_splits}.")


if __name__ == "__main__":
    main()
