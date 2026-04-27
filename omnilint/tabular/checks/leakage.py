"""Leakage detection: correlation-based."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any
from scipy.stats import spearmanr, chi2_contingency


def run(df: pd.DataFrame, target_col: str) -> list[dict]:
    """Run leakage detection checks."""
    issues = []

    issues.extend(check_numeric_leakage(df, target_col))
    issues.extend(check_categorical_leakage(df, target_col))

    return issues


def check_numeric_leakage(df: pd.DataFrame, target_col: str) -> list[dict]:
    """Check numeric features for high correlation with target."""
    issues = []
    if target_col not in df.columns:
        return issues

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target = df[target_col]

    for col in numeric_cols:
        if col == target_col:
            continue
        try:
            corr = df[col].corr(target)
            if pd.isna(corr):
                continue
            abs_corr = abs(corr)
            if abs_corr > 0.9:
                severity = "critical"
                detail = f"Pearson correlation with target = {abs_corr:.2f}"
                suggestion = "This feature may be computed after label. Verify or remove."
            elif abs_corr > 0.7:
                severity = "high"
                detail = f"Pearson correlation with target = {abs_corr:.2f}"
                suggestion = "Investigate this feature for potential leakage."
            else:
                continue

            issues.append({
                "check": "leakage_correlation",
                "column": col,
                "severity": severity,
                "detail": detail,
                "suggestion": suggestion,
            })
        except Exception:
            continue
    return issues


def check_categorical_leakage(df: pd.DataFrame, target_col: str) -> list[dict]:
    """Check categorical features using Cramér's V."""
    issues = []
    if target_col not in df.columns:
        return issues

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    target = df[target_col]

    for col in cat_cols:
        if col == target_col:
            continue
        v = cramers_v(df[col], target)
        if v is None:
            continue
        if v > 0.9:
            severity = "critical"
            detail = f"Cramér's V with target = {v:.2f} - near-perfect association"
            suggestion = "This column may be encoding the label. Drop it."
        elif v > 0.7:
            severity = "high"
            detail = f"Cramér's V with target = {v:.2f}"
            suggestion = "Investigate for leakage potential."
        else:
            continue

        issues.append({
            "check": "leakage_categorical",
            "column": col,
            "severity": severity,
            "detail": detail,
            "suggestion": suggestion,
        })
    return issues


def cramers_v(x: pd.Series, y: pd.Series) -> float | None:
    """Calculate Cramér's V association statistic."""
    try:
        confusion = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion)[0]
        n = confusion.sum().sum()
        min_dim = min(confusion.shape[0], confusion.shape[1]) - 1
        if min_dim == 0:
            return None
        return np.sqrt(chi2 / (n * min_dim))
    except Exception:
        return None