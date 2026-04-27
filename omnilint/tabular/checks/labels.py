"""Label auditing: class imbalance and rare class detection."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


def run(df: pd.DataFrame, target_col: str) -> list[dict]:
    """Run label checks."""
    issues = []

    issues.extend(check_imbalance(df, target_col))
    issues.extend(check_rare_classes(df, target_col))

    return issues


def check_imbalance(df: pd.DataFrame, target_col: str, threshold: float = 0.2) -> list[dict]:
    """Check for class imbalance."""
    issues = []
    if target_col not in df.columns:
        return issues

    value_counts = df[target_col].value_counts()
    if len(value_counts) < 2:
        return issues

    minority = value_counts.min()
    majority = value_counts.max()
    ratio = minority / majority

    if ratio < threshold:
        class_dist = {k: f"{v/len(df)*100:.1f}%" for k, v in value_counts.items()}
        issues.append({
            "check": "class_imbalance",
            "column": target_col,
            "severity": "high" if ratio < 0.1 else "medium",
            "detail": f"Class distribution: {class_dist}. Ratio = {ratio:.3f}",
            "suggestion": "Use SMOTE, class_weight='balanced', or undersample. Avoid accuracy - use F1/AUC-PR.",
        })
    return issues


def check_rare_classes(df: pd.DataFrame, target_col: str, min_samples: int = 50) -> list[dict]:
    """Check for rare classes."""
    issues = []
    if target_col not in df.columns:
        return issues

    value_counts = df[target_col].value_counts()
    for cls, count in value_counts.items():
        if count < min_samples:
            issues.append({
                "check": "rare_class",
                "column": target_col,
                "severity": "medium",
                "detail": f"Class '{cls}' has only {count} samples - below minimum {min_samples}",
                "suggestion": "Merge into similar class, collect more samples, or exclude from training",
            })
    return issues