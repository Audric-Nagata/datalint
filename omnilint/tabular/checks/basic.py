"""Basic data quality checks."""

from __future__ import annotations

import hashlib
import pandas as pd
from typing import Any


def run(df: pd.DataFrame, threshold: float = 0.05) -> list[dict]:
    """Run all basic checks: missing values, types, constants, duplicates."""
    issues = []

    issues.extend(check_missing(df, threshold))
    issues.extend(check_types(df))
    issues.extend(check_constants(df))
    issues.extend(check_exact_duplicates(df))

    return issues


def check_missing(df: pd.DataFrame, threshold: float = 0.05) -> list[dict]:
    """Check for missing values per column."""
    issues = []
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df)
        if missing_pct > threshold:
            issues.append({
                "check": "missing_values",
                "column": col,
                "severity": "high" if missing_pct > 0.3 else "medium",
                "detail": f"{missing_pct*100:.1f}% of values are missing",
                "suggestion": "Impute with median/mean, or add a binary indicator column",
            })
    return issues


def check_types(df: pd.DataFrame) -> list[dict]:
    """Check for data type mismatches."""
    issues = []
    for col in df.columns:
        dtype = df[col].dtype
        if df[col].dtype == "object":
            try:
                pd.to_numeric(df[col])
                issues.append({
                    "check": "type_mismatch",
                    "column": col,
                    "severity": "medium",
                    "detail": f"Column contains strings that could be numeric",
                    "suggestion": "Convert to numeric type for better analysis",
                })
            except (ValueError, TypeError):
                pass
    return issues


def check_constants(df: pd.DataFrame) -> list[dict]:
    """Check for constant (zero variance) columns."""
    issues = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique == 1:
            issues.append({
                "check": "constant_column",
                "column": col,
                "severity": "low",
                "detail": f"All {len(df)} rows have the same value - zero variance",
                "suggestion": "Drop this column; it adds no information",
            })
    return issues


def check_exact_duplicates(df: pd.DataFrame) -> list[dict]:
    """Check for exact duplicate rows using SHA-256 hashing."""
    issues = []
    if df.duplicated().sum() > 0:
        dup_count = df.duplicated().sum()
        dup_pct = dup_count / len(df)
        issues.append({
            "check": "exact_duplicates",
            "severity": "high" if dup_pct > 0.05 else "medium",
            "detail": f"{dup_count} duplicate rows found ({dup_pct*100:.2f}%)",
            "suggestion": "Deduplicate before train/test split to avoid split leakage",
        })
    return issues