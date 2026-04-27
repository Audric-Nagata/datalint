"""Pixel-level statistics utilities for image quality checks."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class PixelStats:
    """Container for pixel statistics of an image."""
    mean_brightness: float
    std_contrast: float
    r_mean: float
    g_mean: float
    b_mean: float
    r_std: float
    g_std: float
    b_std: float


def compute_stats(image_path: Path) -> PixelStats | None:
    """Compute pixel statistics for an image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PixelStats or None on error
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    
    try:
        with Image.open(image_path) as im:
            if im.mode == "L":
                arr = np.array(im)
                return PixelStats(
                    mean_brightness=float(arr.mean()),
                    std_contrast=float(arr.std()),
                    r_mean=float(arr.mean()),
                    g_mean=float(arr.mean()),
                    b_mean=float(arr.mean()),
                    r_std=float(arr.std()),
                    g_std=float(arr.std()),
                    b_std=float(arr.std()),
                )
            elif im.mode == "RGB":
                arr = np.array(im)
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                gray = np.mean(arr, axis=2)
                return PixelStats(
                    mean_brightness=float(gray.mean()),
                    std_contrast=float(gray.std()),
                    r_mean=float(r.mean()),
                    g_mean=float(g.mean()),
                    b_mean=float(b.mean()),
                    r_std=float(r.std()),
                    g_std=float(g.std()),
                    b_std=float(b.std()),
                )
            elif im.mode == "RGBA":
                arr = np.array(im)
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                gray = np.mean(arr[:, :, :3], axis=2)
                return PixelStats(
                    mean_brightness=float(gray.mean()),
                    std_contrast=float(gray.std()),
                    r_mean=float(r.mean()),
                    g_mean=float(g.mean()),
                    b_mean=float(b.mean()),
                    r_std=float(r.std()),
                    g_std=float(g.std()),
                    b_std=float(b.std()),
                )
    except Exception:
        return None


def compute_laplacian_variance(image_path: Path) -> float | None:
    """Compute Laplacian variance for blur detection.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Laplacian variance (higher = sharper)
    """
    try:
        import cv2
    except ImportError:
        return None
    
    try:
        arr = cv2.imread(str(image_path))
        if arr is None:
            return None
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    except Exception:
        return None


def compute_brightness(image_path: Path) -> float | None:
    """Compute mean brightness of an image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Mean pixel intensity (0-255)
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    
    try:
        with Image.open(image_path) as im:
            arr = np.array(im.convert("L"))
            return float(arr.mean())
    except Exception:
        return None


def compute_contrast(image_path: Path) -> float | None:
    """Compute standard deviation of pixel intensities (contrast).
    
    Args:
        image_path: Path to image file
    
    Returns:
        Standard deviation (0-255)
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    
    try:
        with Image.open(image_path) as im:
            arr = np.array(im.convert("L"))
            return float(arr.std())
    except Exception:
        return None


def compute_channel_means(image_path: Path) -> dict[str, float] | None:
    """Compute per-channel means for RGB images.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with 'r', 'g', 'b' keys or None on error
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    
    try:
        with Image.open(image_path) as im:
            if im.mode != "RGB":
                return None
            arr = np.array(im)
            return {
                "r": float(arr[:, :, 0].mean()),
                "g": float(arr[:, :, 1].mean()),
                "b": float(arr[:, :, 2].mean()),
            }
    except Exception:
        return None


def batch_stats(
    image_paths: list[Path],
    max_workers: int = 4,
) -> list[PixelStats]:
    """Compute statistics for multiple images in parallel.
    
    Args:
        image_paths: List of image file paths
        max_workers: Number of parallel workers
    
    Returns:
        List of PixelStats (None for failed images)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    stats = [None] * len(image_paths)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_stats, path): i
            for i, path in enumerate(image_paths)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                stats[idx] = future.result()
            except Exception:
                pass
    
    return stats