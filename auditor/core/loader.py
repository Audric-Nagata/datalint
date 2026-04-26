"""Data loading and schema inference."""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class Schema:
    source: str
    rows: int
    columns: int
    schema: dict


def infer_schema(df: pd.DataFrame) -> dict:
    """Infer column types and nullable status."""
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        nullable = df[col].isna().any()
        if pd.api.types.is_numeric_dtype(dtype):
            inferred = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            inferred = "datetime"
        else:
            inferred = "categorical"
        schema[col] = {
            "inferred_type": inferred,
            "nullable": nullable,
        }
    return schema


def load(source: Union[str, Path, pd.DataFrame]) -> tuple[pd.DataFrame, Schema]:
    """Load data from file path or DataFrame and return with schema."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        filename = "dataframe"
    else:
        path = Path(source)
        filename = path.name
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    if df.empty or df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns and 1 row")

    schema = infer_schema(df)
    return df, Schema(
        source=filename,
        rows=len(df),
        columns=len(df.columns),
        schema=schema,
    )


def validate_minimum_requirements(df: pd.DataFrame) -> None:
    """Raise if dataset doesn't meet minimum requirements."""
    if df.empty:
        raise ValueError("Dataset is empty")
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns")