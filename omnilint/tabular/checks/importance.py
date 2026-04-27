"""Feature importance noise detection using RF probe model."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


def run(df: pd.DataFrame, target_col: str, low_threshold: float = 0.001, high_threshold: float = 0.5) -> list[dict]:
    """Run feature importance noise detection using RF probe."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split

    issues = []

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    is_classification = y.dtype == "object" or y.nunique() < 20

    if is_classification:
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

    if len(df) > 100000:
        _, X_sample, _, y_sample = train_test_split(X, y, train_size=10000, stratify=y if is_classification else None, random_state=42)
    else:
        X_sample, y_sample = X, y

    model.fit(X_sample, y_sample)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    total_importance = importances.sum()
    if total_importance > 0:
        normalized = importances / total_importance
    else:
        normalized = importances

    for col, imp in normalized.items():
        if imp > high_threshold:
            issues.append({
                "check": "dominant_feature",
                "column": col,
                "severity": "critical",
                "detail": f"Single feature holds {imp*100:.1f}% of total importance - abnormal dominance",
                "suggestion": "Cross-check with leakage module. If confirmed, remove before training.",
            })
        elif imp < low_threshold:
            issues.append({
                "check": "useless_feature",
                "column": col,
                "severity": "low",
                "detail": f"Feature importance = {imp:.4f} - below noise floor",
                "suggestion": "Safe to drop. Reduces dimensionality without impacting model.",
            })

    return issues