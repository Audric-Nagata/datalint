"""Image distribution checks."""

from __future__ import annotations

import numpy as np
from PIL import Image
from omnilint.core.loader import ImageDataset


def run(dataset: ImageDataset) -> list[dict]:
    """Run image distribution checks."""
    issues = []
    issues.extend(check_brightness(dataset))
    issues.extend(check_contrast(dataset))
    issues.extend(check_channel_balance(dataset))
    return issues


def check_brightness(dataset: ImageDataset) -> list[dict]:
    """Check for brightness outliers."""
    issues = []
    brightness_values = []
    
    for img in dataset.images:
        if not img.file_path.exists():
            continue
        try:
            with Image.open(img.file_path) as im:
                arr = np.array(im.convert("L"))
                brightness_values.append(arr.mean())
        except Exception:
            pass
    
    if len(brightness_values) < 10:
        return issues
    
    import numpy as np
    p5 = np.percentile(brightness_values, 5)
    p95 = np.percentile(brightness_values, 95)
    
    low_count = sum(1 for b in brightness_values if b < p5)
    high_count = sum(1 for b in brightness_values if b > p95)
    
    if low_count > len(brightness_values) * 0.05:
        issues.append({
            "check": "brightness_outliers",
            "severity": "medium",
            "detail": f"{low_count} underexposed images (below P5)",
            "suggestion": "Apply histogram equalization or check camera calibration.",
        })
    
    if high_count > len(brightness_values) * 0.05:
        issues.append({
            "check": "brightness_outliers",
            "severity": "medium",
            "detail": f"{high_count} overexposed images (above P95)",
            "suggestion": "Check for camera overexposure issues.",
        })
    
    return issues


def check_contrast(dataset: ImageDataset) -> list[dict]:
    """Check for near-zero contrast (blank) images."""
    issues = []
    low_contrast = []
    
    for img in dataset.images:
        if not img.file_path.exists():
            continue
        try:
            from PIL import Image
            with Image.open(img.file_path) as im:
                arr = np.array(im.convert("L"))
                std = arr.std()
                if std < 5:
                    low_contrast.append(img.filename)
        except Exception:
            pass
    
    if low_contrast:
        pct = len(low_contrast) / len(dataset.images) * 100
        issues.append({
            "check": "low_contrast",
            "severity": "critical",
            "detail": f"{len(low_contrast)} near-blank images ({pct:.1f}%)",
            "suggestion": "Remove blank/near-uniform images.",
            "asset": low_contrast[:5],
        })
    
    return issues


def check_channel_balance(dataset: ImageDataset) -> list[dict]:
    """Check for color channel imbalance."""
    issues = []
    
    r_means = []
    g_means = []
    b_means = []
    
    for img in dataset.images:
        if not img.file_path.exists():
            continue
        try:
            from PIL import Image
            with Image.open(img.file_path) as im:
                if im.mode == "RGB":
                    arr = np.array(im)
                    r_means.append(arr[:, :, 0].mean())
                    g_means.append(arr[:, :, 1].mean())
                    b_means.append(arr[:, :, 2].mean())
        except Exception:
            pass
    
    if len(r_means) < 10:
        return issues
    
    import numpy as np
    r_mean = np.mean(r_means)
    g_mean = np.mean(g_means)
    b_mean = np.mean(b_means)
    
    max_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(b_mean - r_mean))
    
    if max_diff > 0.3:
        issues.append({
            "check": "channel_imbalance",
            "severity": "low",
            "detail": f"Channel means: R={r_mean:.2f}, G={g_mean:.2f}, B={b_mean:.2f}. Max diff={max_diff:.2f}",
            "suggestion": "Apply per-channel normalization before training.",
        })
    
    return issues