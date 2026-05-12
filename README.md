# LSC2TEXT

## Requirements

- uv (Project and package manager)

## Data setup

Save the LSC70 (Available in https://data.mendeley.com/datasets/9ssyn8tff5/2) in the `/data/raw` folder. Then execute the CSV generation script by executing `uv run .\scripts\generate_dataset_csv.py`.

## Testing

Test can be run with `uv run python -m unittest discover -s tests -p "test*.py"`.

## Feature Optimization

Run the Optuna-based HOG+LBP optimization script:

```bash
uv run python -m scripts.optimize_features --n-trials 30 --top-k 5
```

Outputs are written to `artifacts/experiments/feature_optimization/`.

## Inference

Run single-image inference with a trained registry model:

```bash
uv run python -m scripts.infer --image .\path\to\image.jpg
```

Use `--model-path` to point to a different `model.joblib` and `--top-k` to change how many predictions are returned:

```bash
uv run python -m scripts.infer --image .\path\to\image.jpg --model-path models/registry/svm/direct-svm-20260511-220859/model.joblib --top-k 3
```

The command prints a JSON payload to stdout with the model path, timestamp, and ranked predictions.

## FastAPI Backend

Start the local API server:

```bash
uv run python -m src.api.api --model-path models/registry/svm/direct-svm-20260511-220859/model.joblib --port 8000
```

Available endpoints:

- `GET /health` returns readiness information
- `GET /metadata` returns the service description and default model path
- `POST /predict` accepts one uploaded image and returns ranked predictions as JSON

The API uses the same inference pipeline as `scripts/infer.py` and is intended for local development and testing.
