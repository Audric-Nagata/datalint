"""Perceptual hashing utilities for image duplicate detection."""

from __future__ import annotations

from pathlib import Path
from typing import Literal


def compute_phash(image_path: Path) -> str | None:
    """Compute perceptual hash for an image file.
    
    Returns hex string or None if image cannot be decoded.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return None
    
    try:
        with Image.open(image_path) as im:
            return str(imagehash.phash(im))
    except Exception:
        return None


def find_exact_duplicates(
    image_paths: list[Path],
    parallel: bool = True,
) -> dict[str, list[str]]:
    """Find exact duplicate images by perceptual hash.
    
    Args:
        image_paths: List of image file paths
        parallel: Use parallel processing for large datasets
    
    Returns:
        Dict mapping hash -> list of duplicate filenames
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return {}
    
    from collections import defaultdict
    hash_map = defaultdict(list)
    
    for path in image_paths:
        try:
            with Image.open(path) as im:
                h = imagehash.phash(im)
                hash_map[str(h)].append(path.name)
        except Exception:
            continue
    
    return {h: files for h, files in hash_map.items() if len(files) > 1}


def hash_to_faiss_index(
    image_paths: list[Path],
    index_path: Path | None = None,
) -> tuple[list[str], object]:
    """Build FAISS index from perceptual hashes for fast similarity search.
    
    Args:
        image_paths: List of image file paths
        index_path: Optional path to saveFAISS index
    
    Returns:
        (hash_strings, faiss_index) tuple
    """
    try:
        import imagehash
        import faiss
        from PIL import Image
    except ImportError:
        return [], None
    
    hashes = []
    filenames = []
    
    for path in image_paths:
        try:
            with Image.open(path) as im:
                h = imagehash.phash(im)
                hashes.append(h)
                filenames.append(path.name)
        except Exception:
            continue
    
    if not hashes:
        return [], None
    
    hash_matrix = imagehash.hash_to_faiss(hashes)
    index = faiss.IndexFlat(64)
    index.add(hash_matrix)
    
    if index_path:
        faiss.write_index(index, str(index_path))
    
    return filenames, index


def find_similar_by_hamming(
    h1: str,
    h2: str,
    threshold: int = 5,
) -> bool:
    """Check if two perceptual hashes are similar by Hamming distance.
    
    Args:
        h1: First hash (hex string)
        h2: Second hash (hex string)
        threshold: Max Hamming distance (default: 5)
    
    Returns:
        True if hashes are within threshold
    """
    try:
        import imagehash
    except ImportError:
        return False
    
    h1_hash = imagehash.ImageHash(h1)
    h2_hash = imagehash.ImageHash(h2)
    
    return (h1_hash - h2_hash) <= threshold