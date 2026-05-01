# LSC2TEXT

## Requirements

- uv (Project and package manager)

## Data setup

Save the LSC70 (Available in https://data.mendeley.com/datasets/9ssyn8tff5/2) in the `/data/raw` folder. Then execute the CSV generation script by executing `uv run .\scripts\generate_dataset_csv.py`.

## Testing

Test can be run with `uv run python -m unittest discover -s tests -p "test*.py"`. 