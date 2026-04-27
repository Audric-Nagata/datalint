"""Tests for image duplicate checks."""

from __future__ import annotations

import pytest
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

from auditor.image.checks import duplicates
from auditor.core.loader import ImageDataset, ImageEntry


pytestmark = pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")


def test_check_exact_duplicates_none(sample_image_dataset: ImageDataset, tmp_path: Path):
    """Test with no duplicates."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    for i in range(5):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i}.jpg")
    
    for img in sample_image_dataset.images:
        img.file_path = img_dir / img.filename
    
    issues = duplicates.check_exact_duplicates(sample_image_dataset)
    assert len(issues) == 0


def test_check_exact_duplicates_found(sample_image_dataset: ImageDataset, tmp_path: Path):
    """Test when duplicates are found."""
    if not HAS_IMAGEHASH:
        pytest.skip("imagehash not installed")
    
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    for i in range(3):
        Image.fromarray(arr).save(img_dir / f"dup_{i}.jpg")
    
    for img in sample_image_dataset.images:
        img.file_path = img_dir / "dup_0.jpg"
    
    issues = duplicates.check_exact_duplicates(sample_image_dataset)
    assert len(issues) >= 1


def test_check_exact_duplicates_library_missing(sample_image_dataset: ImageDataset):
    """Test when imagehash library is not installed."""
    if HAS_IMAGEHASH:
        pytest.skip("imagehash is installed")
    
    issues = duplicates.check_exact_duplicates(sample_image_dataset)
    assert len(issues) == 1
    assert "imagehash" in issues[0]["detail"]


def test_run_duplicates_all(sample_image_dataset: ImageDataset):
    """Test run function includes all checks."""
    issues = duplicates.run(sample_image_dataset)
    assert isinstance(issues, list)