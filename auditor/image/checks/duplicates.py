"""Image duplicate checks."""

from __future__ import annotations

from auditor.core.loader import ImageDataset


def run(dataset: ImageDataset) -> list[dict]:
    """Run image duplicate checks."""
    issues = []
    issues.extend(check_exact_duplicates(dataset))
    return issues


def check_exact_duplicates(dataset: ImageDataset) -> list[dict]:
    """Check for exact duplicate images using perceptual hash."""
    issues = []
    
    try:
        import imagehash
        from PIL import Image
        from collections import defaultdict
    except ImportError:
        return [{
            "check": "duplicates",
            "severity": "low",
            "detail": "imagehash library not installed. Install with: pip install imagehash",
            "suggestion": "pip install imagehash",
        }]
    
    hashes = defaultdict(list)
    
    for img in dataset.images:
        if not img.file_path.exists():
            continue
        try:
            with Image.open(img.file_path) as im:
                h = imagehash.phash(im)
                hashes[h].append(img.filename)
        except Exception:
            pass
    
    dup_pairs = 0
    for h, files in hashes.items():
        if len(files) > 1:
            dup_pairs += len(files) - 1
    
    if dup_pairs > 0:
        pct = dup_pairs / len(dataset.images) * 100
        issues.append({
            "check": "exact_duplicates",
            "severity": "high",
            "detail": f"{dup_pairs} exact duplicate images ({pct:.1f}%)",
            "suggestion": "Remove duplicates before training.",
        })
    
    return issues