"""Loading and splitting the HR attrition table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONSTANT_COLUMNS, DATA_PATH, ID_COLUMN, SEED, TARGET, TEST_SIZE


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the csv, drop constant columns and the id, and turn the target into 0/1."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.drop(columns=[c for c in CONSTANT_COLUMNS + [ID_COLUMN] if c in df.columns])
    df[TARGET] = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names. Ordinal survey scores stay numeric."""
    cols = [c for c in df.columns if c != TARGET]
    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in cols if c not in categorical]
    return numeric, categorical


def split(df: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split. Rows are independent employees, so a plain split is honest here."""
    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[TARGET])
    if columns is not None:
        X = X[columns]
    return X, df[TARGET].to_numpy()


def survival_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Event flag and duration in years. Whole-year tenure is shifted by half a year so nobody has zero time."""
    event = df[TARGET].to_numpy().astype(bool)
    time = df["YearsAtCompany"].to_numpy().astype(float) + 0.5
    return event, time
