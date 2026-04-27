"""CLIP embedding wrapper for image similarity and zero-shot classification."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


_MODEL_CACHE = None


def load_model(
    model_name: str = "ViT-B/32",
    device: str | None = None,
) -> object:
    """Load CLIP model with lazy caching.
    
    Args:
        model_name: CLIP model variant (default: ViT-B/32)
        device: 'cuda', 'cpu', or None for auto-detect
    
    Returns:
        Loaded CLIP model
    """
    global _MODEL_CACHE
    
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    try:
        import torch
        import clip
    except ImportError:
        raise ImportError("torch and clip are required. Install: pip install torch clip")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    
    _MODEL_CACHE = (model, preprocess, device)
    return model, preprocess, device


def encode_image(
    image_path: Path,
    model_name: str = "ViT-B/32",
) -> np.ndarray | None:
    """Encode a single image to CLIP embedding.
    
    Args:
        image_path: Path to image file
        model_name: CLIP model variant
    
    Returns:
        Embedding vector (512-dim for ViT-B/32) or None on error
    """
    try:
        import torch
        import clip
        from PIL import Image
    except ImportError:
        return None
    
    model, preprocess, device = load_model(model_name)
    
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
        
        return embedding.cpu().numpy().flatten()
    except Exception:
        return None


def encode_images_batch(
    image_paths: list[Path],
    model_name: str = "ViT-B/32",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode multiple images to CLIP embeddings in batches.
    
    Args:
        image_paths: List of image file paths
        model_name: CLIP model variant
        batch_size: Batch size for encoding
    
    Returns:
        Embedding matrix (N x 512)
    """
    try:
        import torch
        import clip
        from PIL import Image
    except ImportError:
        return np.array([])
    
    model, preprocess, device = load_model(model_name)
    
    embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                tensor = preprocess(image)
                batch_tensors.append(tensor)
            except Exception:
                continue
        
        if not batch_tensors:
            continue
        
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            batch_embeddings = model.encode_image(batch)
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    if embeddings:
        return np.vstack(embeddings)
    return np.array([])


def encode_text(
    texts: list[str],
    model_name: str = "ViT-B/32",
) -> np.ndarray:
    """Encode text labels to CLIP text embeddings.
    
    Args:
        texts: List of text strings (class names)
        model_name: CLIP model variant
    
    Returns:
        Embedding matrix (len(texts) x 512)
    """
    try:
        import torch
        import clip
    except ImportError:
        return np.array([])
    
    model, _, device = load_model(model_name)
    
    try:
        text_tokens = clip.tokenize(texts).to(device)
        
        with torch.no_grad():
            embeddings = model.encode_text(text_tokens)
        
        return embeddings.cpu().numpy()
    except Exception:
        return np.array([])


def zero_shot_classify(
    image_embedding: np.ndarray,
    text_embeddings: np.ndarray,
    texts: list[str],
) -> tuple[str, float]:
    """Classify image by comparing to text label embeddings.
    
    Args:
        image_embedding: Image embedding vector
        text_embeddings: Text embeddings matrix
        texts: List of class label strings
    
    Returns:
        (predicted_label, confidence_score)
    """
    if image_embedding.size == 0 or text_embeddings.size == 0:
        return "", 0.0
    
    image_embedding = image_embedding / np.linalg.norm(image_embedding)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    similarities = text_embeddings @ image_embedding
    
    best_idx = np.argmax(similarities)
    return texts[best_idx], float(similarities[best_idx])


def find_near_duplicates(
    embeddings: np.ndarray,
    threshold: float = 0.97,
) -> list[tuple[int, int, float]]:
    """Find near-duplicate pairs by cosine similarity.
    
    Args:
        embeddings: Embedding matrix (N x D)
        threshold: Cosine similarity threshold
    
    Returns:
        List of (idx1, idx2, similarity) tuples
    """
    if embeddings.size == 0:
        return []
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    similarity_matrix = normalized @ normalized.T
    
    n = len(embeddings)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            if sim >= threshold:
                pairs.append((i, j, sim))
    
    return pairs