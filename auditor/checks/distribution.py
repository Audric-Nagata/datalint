"""Distribution analysis: skewness and outlier detection."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


def run(df: pd.DataFrame) -> list[dict]:
    """Run distribution analysis checks."""
    issues = []

    issues.extend(check_skewness(df))
    issues.extend(check_outliers_iqr(df))
    issues.extend(check_outliers_zscore(df))

    return issues


def check_skewness(df: pd.DataFrame, threshold: float = 1.0) -> list[dict]:
    """Detect skewed numeric distributions."""
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        skew = df[col].dropna().skew()
        if abs(skew) > threshold:
            issues.append({
                "check": "skewness",
                "column": col,
                "severity": "medium",
                "detail": f"Skewness = {abs(skew):.2f} ({'right' if skew > 0 else 'left'}-skewed)",
                "suggestion": "Apply log1p transform or Box-Cox to normalize distribution",
            })
    return issues


def check_outliers_iqr(df: pd.DataFrame) -> list[dict]:
    """Detect outliers using IQR method."""
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            outlier_pct = outlier_count / len(df)
            issues.append({
                "check": "outliers_iqr",
                "column": col,
                "severity": "medium" if outlier_pct < 0.05 else "high",
                "detail": f"{outlier_count} outlier rows ({outlier_pct*100:.2f}%) via IQR",
                "suggestion": "Inspect outliers manually; consider winsorizing at 1st/99th percentile",
            })
    return issues


def check_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> list[dict]:
    """Detect outliers using Z-score method."""
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            continue

        z_scores = np.abs((df[col] - mean) / std)
        outlier_mask = z_scores > threshold
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            outlier_pct = outlier_count / len(df)
            issues.append({
                "check": "outliers_zscore",
                "column": col,
                "severity": "medium" if outlier_pct < 0.05 else "high",
                "detail": f"{outlier_count} outlier rows ({outlier_pct*100:.2f}%) via Z-score",
                "suggestion": "Review these values; they may be data entry errors",
            })
    return issues


def merge_outlier_flags(iqr_flags: pd.Series, z_flags: pd.Series) -> pd.Series:
    """Merge IQR and Z-score outlier flags."""
    return iqr_flags | z_flags