"""Duplicate and near-duplicate detection."""

from __future__ import annotations

import hashlib
import pandas as pd
import numpy as np
from typing import Any


def run(df: pd.DataFrame, threshold: float = 0.95) -> list[dict]:
    """Run duplicate detection checks."""
    issues = []

    issues.extend(find_exact_duplicates(df))

    if len(df) < 10000:
        issues.extend(find_near_duplicates_small(df, threshold))
    else:
        issues.extend(find_near_duplicates_large(df, threshold))

    return issues


def find_exact_duplicates(df: pd.DataFrame) -> list[dict]:
    """Find exact duplicate rows using row hashing."""
    issues = []
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        dup_pct = dup_count / len(df)
        issues.append({
            "check": "exact_duplicates",
            "severity": "high" if dup_pct > 0.05 else "medium",
            "detail": f"{dup_count} exact duplicate rows ({dup_pct*100:.2f}%)",
            "suggestion": "Deduplicate before train/test split to avoid split leakage",
        })
    return issues


def find_near_duplicates_small(df: pd.DataFrame, threshold: float = 0.95) -> list[dict]:
    """Brute-force near-duplicate detection for small datasets."""
    issues = []
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return issues

    normalized = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-8)

    norms = np.linalg.norm(normalized.values, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = normalized.values / norms

    dot_products = normalized @ normalized.T
    similarities = np.triu(dot_products, k=1)

    pairs = np.argwhere(similarities > threshold)
    if len(pairs) > 0:
        issues.append({
            "check": "near_duplicates",
            "severity": "medium",
            "detail": f"{len(pairs)} near-duplicate pairs found (cosine similarity > {threshold})",
            "suggestion": "Review pairs for potential duplicate records",
        })
    return issues


def find_near_duplicates_large(df: pd.DataFrame, threshold: float = 0.95) -> list[dict]:
    """FAISS-based near-duplicate detection for large datasets."""
    issues = []
    try:
        import faiss
    except ImportError:
        return issues

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return issues

    normalized = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-8)
    vectors = normalized.values.astype(np.float32)

    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    k = 2
    distances, indices = index.search(vectors, k)

    pairs = np.argwhere(distances[:, 1] > threshold)
    if len(pairs) > 0:
        issues.append({
            "check": "near_duplicates",
            "severity": "medium",
            "detail": f"{len(pairs)} near-duplicate pairs found via FAISS (similarity > {threshold})",
            "suggestion": "Review pairs for potential duplicate records",
        })
    return issues