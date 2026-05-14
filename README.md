# LSC2TEXT

## Details

For details on the project and my motivation on doing this (though it is only available in Spanish, for the moment): [PROYECTO](PROYECTO.MD)

## How to Set Up this Project?

### 1. Requirements

You need to download the following: 

- uv (Project and package manager)

### 2. Set Up the Data

1. Save the LSC70 (Available in https://data.mendeley.com/datasets/9ssyn8tff5/2) in the `/data/raw` folder. 
2. Execute the CSV generation script by executing `uv run .\scripts\generate_dataset_csv.py`.
3. Execute the filtering script by executing `uv run .\scripts\filter_dataset.py`.
4. Split the filtered dataset into train/valid: `uv run .\scripts\split_dataset.py`.

This can be automatically done with default values through the shell with `setup.sh`.

### 3. Train the Model

Train the model by using `uv run .\scripts\train.py`.

### 4. Start the Backend

Start the local API server:

```bash
uv run python -m src.api.api --model-path models/registry/svm/direct-svm-20260511-220859/model.joblib --port 8000
```

Available endpoints:

- `GET /health` returns readiness information
- `GET /metadata` returns the service description and default model path
- `POST /predict` accepts one uploaded image and returns ranked predictions as JSON

The API uses the same inference pipeline as `scripts/infer.py` and is intended for local development and testing.

### 5. Access the Frontend

Visit `http://127.0.0.1:8000/` after (4) to upload a single image and view the rendered prediction result.

The frontend uses server-rendered templates and is designed to work locally with the backend API.

## Testing

Test can be run with `uv run python -m unittest discover -s tests -p "test*.py"`.



