# /// script
# dependencies =[
# ]
# ///

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
FRAME_RE = re.compile(r"_(\d+)$")


def normalize_sign(sign_folder: str) -> str:
    if sign_folder == "NN":
        return "Ñ"
    if sign_folder == "ANNOS":
        return "AÑOS"
    if sign_folder == "MILLON":
        return "MILLÓN"
    return sign_folder


def parse_frame(filename_stem: str) -> str:
    match = FRAME_RE.search(filename_stem)
    if match:
        return match.group(1)
    return ""


def iter_image_files(root: Path) -> Iterable[Path]:
    for section_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for participant_dir in sorted(p for p in section_dir.iterdir() if p.is_dir()):
            for sign_dir in sorted(p for p in participant_dir.iterdir() if p.is_dir()):
                for image_path in sorted(
                    p for p in sign_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    yield image_path


def build_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for image_path in iter_image_files(root):
        sign_dir = image_path.parent
        participant_dir = sign_dir.parent

        sign = normalize_sign(sign_dir.name)
        participant = participant_dir.name
        frame = parse_frame(image_path.stem)

        relative_path = image_path.relative_to(root).as_posix()

        rows.append(
            {
                "Signo": sign,
                "Participante": participant,
                "Frame": frame,
                "Path": relative_path,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a CSV from data/raw with Signo, Participante, Frame, and Path."
    )
    parser.add_argument(
        "--root",
        default=Path("data/raw/LSC70"),
        type=Path,
        help="Root folder that contains sections (default: data/raw/LSC70).",
    )
    parser.add_argument(
        "--output",
        default=Path("data/raw/dataset.csv"),
        type=Path,
        help="Output CSV path (default: data/raw/dataset.csv).",
    )
    args = parser.parse_args()

    root = args.root
    output_path = args.output

    if not root.exists():
        raise SystemExit(f"Root folder not found: {root}")

    rows = build_rows(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Signo", "Participante", "Frame", "Path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
