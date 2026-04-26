"""Sentence-transformer embeddings wrapper for text vectorization."""

from __future__ import annotations

import numpy as np
from typing import Any

import pandas as pd


_model = None


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load and cache sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
    return _model


def encode(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts to dense vectors."""
    model = load_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)


def encode_mixed_row(row: pd.Series, schema: dict) -> np.ndarray:
    """Concatenate normalized numerics + embedded text into single vector."""
    vectors = []
    for col, val in row.items():
        if schema.get(col, {}).get("inferred_type") == "numeric":
            vectors.append(float(val) if pd.notna(val) else 0.0)
        else:
            text = str(val) if pd.notna(val) else ""
            if text:
                vec = encode([text])
                vectors.extend(vec[0])
            else:
                vectors.extend([0.0] * 384)
    return np.array(vectors)