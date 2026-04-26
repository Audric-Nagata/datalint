"""Statistical utility functions."""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Calculate IQR-based outlier bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def zscore_flags(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean outlier mask based on Z-score."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramér's V association statistic."""
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = stats.chi2_contingency(confusion)
    n = confusion.sum().sum()
    min_dim = min(confusion.shape[0], confusion.shape[1]) - 1
    if min_dim == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def skewness(series: pd.Series) -> float:
    """Calculate skewness coefficient."""
    return series.dropna().skew()


def correlation(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    """Calculate correlation between two series."""
    if method == "pearson":
        return x.corr(y)
    elif method == "spearman":
        return x.corr(y, method="spearman")
    return x.corr(y)