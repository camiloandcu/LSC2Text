# ///script
# dependencies =[
# ]
# ///

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def count_by_section_sign_from_csv(dataset_csv: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    with dataset_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path_value = (row.get("Path") or "").strip()
            sign = (row.get("Signo") or "").strip()
            if not path_value:
                continue

            section = Path(path_value).parts[0]
            key = (section, sign)
            counts[key] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count images per section and sign from the dataset CSV."
    )
    parser.add_argument(
        "--input",
        default=Path("data/raw/dataset.csv"),
        type=Path,
        help="Input CSV path (default: data/raw/dataset.csv).",
    )
    parser.add_argument(
        "--output",
        default=Path("data/image_counts_by_section_sign.csv"),
        type=Path,
        help="Output CSV path (default: data/image_counts_by_section_sign.csv).",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    counts = count_by_section_sign_from_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Seccion", "Signo", "CantidadImagenes"])
        for (section, sign), total in sorted(counts.items()):
            writer.writerow([section, sign, total])

    print(f"Wrote {len(counts)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
