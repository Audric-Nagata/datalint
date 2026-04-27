"""Image integrity checks."""

from __future__ import annotations

from auditor.core.loader import ImageDataset


def run(dataset: ImageDataset) -> list[dict]:
    """Run all image integrity checks."""
    issues = []
    issues.extend(check_corrupt_files(dataset))
    issues.extend(check_resolution(dataset))
    issues.extend(check_format_consistency(dataset))
    return issues


def check_corrupt_files(dataset: ImageDataset) -> list[dict]:
    """Check for corrupt/unreadable image files."""
    issues = []
    corrupt_count = 0
    corrupt_samples = []
    
    for img in dataset.images:
        if not img.file_path.exists():
            corrupt_count += 1
            corrupt_samples.append(img.filename)
            continue
        
        try:
            with open(img.file_path, "rb") as f:
                header = f.read(16)
                if not header:
                    corrupt_count += 1
                    corrupt_samples.append(img.filename)
        except Exception:
            corrupt_count += 1
            corrupt_samples.append(img.filename)
    
    if corrupt_count > 0:
        pct = corrupt_count / len(dataset.images) * 100
        issues.append({
            "check": "corrupt_files",
            "severity": "critical",
            "detail": f"{corrupt_count} corrupt/unreadable images ({pct:.1f}%)",
            "suggestion": "Remove corrupt files from dataset. Check upload/download pipeline.",
            "asset": corrupt_samples[:5] if corrupt_samples else None,
        })
    
    return issues


def check_resolution(dataset: ImageDataset) -> list[dict]:
    """Check for extreme resolution outliers."""
    issues = []
    
    widths = [img.width for img in dataset.images]
    heights = [img.height for img in dataset.images]
    
    if not widths or not heights:
        return issues
    
    import statistics
    med_w = statistics.median(widths)
    med_h = statistics.median(heights)
    
    outliers = []
    for img in dataset.images:
        if img.width < 32 or img.height < 32:
            outliers.append(img.filename)
        elif img.width > med_w * 4 or img.height > med_h * 4:
            outliers.append(img.filename)
    
    if outliers:
        pct = len(outliers) / len(dataset.images) * 100
        issues.append({
            "check": "resolution_outlier",
            "severity": "medium",
            "detail": f"{len(outliers)} images with extreme resolution ({pct:.1f}%)",
            "suggestion": "Verify these samples. May cause issues with fixed-size model inputs.",
            "asset": outliers[:5] if outliers else None,
        })
    
    return issues


def check_format_consistency(dataset: ImageDataset) -> list[dict]:
    """Check for mixed image formats."""
    issues = []
    formats = {}
    
    for img in dataset.images:
        ext = img.file_path.suffix.lower()
        formats[ext] = formats.get(ext, 0) + 1
    
    if len(formats) > 1:
        main_format = max(formats, key=formats.get)
        other_formats = [f for f in formats if f != main_format]
        issues.append({
            "check": "format_inconsistency",
            "severity": "low",
            "detail": f"Mixed formats: {formats}. Primary: {main_format}",
            "suggestion": "Standardize to single format for consistent preprocessing.",
        })
    
    return issues