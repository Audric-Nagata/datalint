"""Image label checks."""

from __future__ import annotations

from omnilint.core.loader import ImageDataset


def run(dataset: ImageDataset) -> list[dict]:
    """Run image label checks."""
    issues = []
    issues.extend(check_label_file_mismatch(dataset))
    issues.extend(check_class_imbalance(dataset))
    return issues


def check_label_file_mismatch(dataset: ImageDataset) -> list[dict]:
    """Check for label-file mismatches."""
    issues = []
    
    manifest_names = {img.filename for img in dataset.images}
    disk_names = set()
    
    if dataset.root_path:
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            disk_names.update(f.name for f in dataset.root_path.rglob(f"*{ext}"))
    
    missing_in_manifest = disk_names - manifest_names
    missing_on_disk = manifest_names - disk_names
    
    if missing_in_manifest or missing_on_disk:
        issues.append({
            "check": "label_file_mismatch",
            "severity": "high",
            "detail": f"{len(missing_on_disk)} manifest entries without file. {len(missing_in_manifest)} images without manifest.",
            "suggestion": "Re-sync manifest with image directory.",
        })
    
    return issues


def check_class_imbalance(dataset: ImageDataset) -> list[dict]:
    """Check for class imbalance."""
    issues = []

    if not dataset.categories:
        return []
    
    import pandas as pd
    class_counts = {}
    
    for img in dataset.images:
        for ann in img.annotations:
            cat_id = ann.get("category_id", 0)
            class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
    
    if not class_counts:
        return []
    
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    ratio = min_count / max_count if max_count > 0 else 0
    
    if ratio < 0.2:
        cat_name = dataset.categories[min(class_counts, key=class_counts.get)]
        issues.append({
            "check": "class_imbalance",
            "severity": "high",
            "detail": f"Class imbalance ratio: {ratio:.2f}. Rare class: {cat_name}",
            "suggestion": "Apply class weighting or oversample minority class.",
        })
    
    return issues