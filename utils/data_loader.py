import pandas as pd
from pathlib import Path


SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{suffix}'. Supported: {SUPPORTED_FORMATS}")

    loaders = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
    }
    return loaders[suffix](file_path)
