#!/usr/bin/env bash
set -e  # stop on error

echo "  Starting Colab setup..."

# --- 1. Ensure we're in the repo root ---
if [ ! -f "pyproject.toml" ] && [ ! -d "scripts" ]; then
  echo "Run this script from the root of your project"
  exit 1
fi

# --- 2. Install uv (fast Python runner) ---
echo "  Installing uv..."
pip install -q uv

# --- 3. Create data directories ---
echo "  Creating directories..."
mkdir -p data/raw

# --- 4. Download dataset ---
DATA_URL="https://data.mendeley.com/public-files/datasets/9ssyn8tff5/files/5a25e9cb-d67b-4590-bb90-e1784a90a578/file_downloaded"
ZIP_PATH="data/raw/lsc70.zip"

echo "  Downloading dataset..."
wget -O "$ZIP_PATH" "$DATA_URL"

# --- 5. Unzip dataset ---
echo "  Extracting dataset..."
unzip -q -o "$ZIP_PATH" -d data/raw/

# --- 6. Run scripts using uv ---
echo "  Running dataset generation..."
uv run ./scripts/generate_dataset_csv.py

echo "  Filtering dataset..."
uv run ./scripts/filter_lsc70w_dataset.py

# --- 7. Export Python PATH ---
export PYTHONPATH=.

echo "✅ Setup complete!"