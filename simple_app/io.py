from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_df(df: pd.DataFrame, path: str) -> str:
    ensure_dir(os.path.dirname(path))
    if path.endswith('.csv'):
        df.to_csv(path, index=False)
    elif path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        path = path + '.parquet'
        df.to_parquet(path, index=False)
    return path


def load_df(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file: {path}")

