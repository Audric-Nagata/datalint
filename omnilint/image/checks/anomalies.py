"""Image anomaly checks."""

from __future__ import annotations

from omnilint.core.loader import ImageDataset


def run(dataset: ImageDataset) -> list[dict]:
    """Run image anomaly checks."""
    issues = []
    issues.extend(check_blurriness(dataset))
    issues.extend(check_exposure(dataset))
    return issues


def check_blurriness(dataset: ImageDataset) -> list[dict]:
    """Check for blurry images using Laplacian variance."""
    issues = []
    blurry = []
    
    try:
        import cv2
    except ImportError:
        return [{
            "check": "blurriness",
            "severity": "low",
            "detail": "OpenCV not installed. Install with: pip install opencv-python",
            "suggestion": "pip install opencv-python",
        }]
    
    for img in dataset.images:
        if not img.file_path.exists():
            continue
        try:
            arr = cv2.imread(str(img.file_path))
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                blurry.append(img.filename)
        except Exception:
            pass
    
    if blurry:
        pct = len(blurry) / len(dataset.images) * 100
        issues.append({
            "check": "blurry_image",
            "severity": "medium",
            "detail": f"{len(blurry)} blurry images ({pct:.1f}%)",
            "suggestion": "Remove or recapture blurry samples.",
            "asset": blurry[:5],
        })
    
    return issues


def check_exposure(dataset: ImageDataset) -> list[dict]:
    """Check for overexposed/underexposed images."""
    issues = []
    overexposed = []
    underexposed = []
    
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return issues
    
    for img in dataset.images:
        if not img.file_path.exists():
            continue
        try:
            with Image.open(img.file_path) as im:
                arr = np.array(im.convert("L"))
                mean = arr.mean()
                if mean > 240:
                    overexposed.append(img.filename)
                elif mean < 15:
                    underexposed.append(img.filename)
        except Exception:
            pass
    
    if overexposed:
        pct = len(overexposed) / len(dataset.images) * 100
        issues.append({
            "check": "overexposed",
            "severity": "medium",
            "detail": f"{len(overexposed)} overexposed images ({pct:.1f}%)",
            "suggestion": "Check camera exposure settings.",
            "asset": overexposed[:5],
        })
    
    if underexposed:
        pct = len(underexposed) / len(dataset.images) * 100
        issues.append({
            "check": "underexposed",
            "severity": "medium",
            "detail": f"{len(underexposed)} underexposed images ({pct:.1f}%)",
            "suggestion": "Improve lighting or adjust camera settings.",
            "asset": underexposed[:5],
        })
    
    return issues