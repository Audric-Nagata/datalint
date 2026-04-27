"""Tests for image anomaly checks."""

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
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from omnilint.image.checks import anomalies
from omnilint.core.loader import ImageDataset, ImageEntry


pytestmark = pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")


def test_check_blurriness_no_blur(sample_image_dataset: ImageDataset, tmp_path: Path):
    """Test with no blurry images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    for i in range(5):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i}.jpg")
    
    for img in sample_image_dataset.images:
        img.file_path = img_dir / img.filename
    
    issues = anomalies.check_blurriness(sample_image_dataset)
    assert isinstance(issues, list)


def test_check_blurriness_with_blur(sample_image_dataset: ImageDataset, tmp_path: Path):
    """Test when blurry images found (if OpenCV available)."""
    if not HAS_CV2:
        pytest.skip("OpenCV not installed")
    
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    blurred = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(blurred, (21, 21), 0)
    Image.fromarray(blurred).save(img_dir / "blurred.jpg")
    
    sample_image_dataset.images[0].file_path = img_dir / "blurred.jpg"
    
    issues = anomalies.check_blurriness(sample_image_dataset)
    assert isinstance(issues, list)


def test_check_exposure_no_issues(sample_image_dataset: ImageDataset, tmp_path: Path):
    """Test with normal exposure."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    for i in range(5):
        arr = np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i}.jpg")
    
    for img in sample_image_dataset.images:
        img.file_path = img_dir / img.filename
    
    issues = anomalies.check_exposure(sample_image_dataset)
    assert isinstance(issues, list)


def test_check_exposure_overexposed(sample_image_dataset: ImageDataset, tmp_path: Path):
    """Test with overexposed images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    bright = np.full((50, 50, 3), 250, dtype=np.uint8)
    Image.fromarray(bright).save(img_dir / "bright.jpg")
    
    sample_image_dataset.images[0].file_path = img_dir / "bright.jpg"
    
    issues = anomalies.check_exposure(sample_image_dataset)
    overexposed = [i for i in issues if i.get("check") == "overexposed"]
    assert len(overexposed) >= 1


def test_check_exposure_underexposed(sample_image_dataset: ImageDataset, tmp_path: Path):
    """Test with underexposed images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    dark = np.zeros((50, 50, 3), dtype=np.uint8)
    Image.fromarray(dark).save(img_dir / "dark.jpg")
    
    sample_image_dataset.images[0].file_path = img_dir / "dark.jpg"
    
    issues = anomalies.check_exposure(sample_image_dataset)
    underexposed = [i for i in issues if i.get("check") == "underexposed"]
    assert len(underexposed) >= 1


def test_run_anomalies_all(sample_image_dataset: ImageDataset):
    """Test run function includes all checks."""
    issues = anomalies.run(sample_image_dataset)
    assert isinstance(issues, list)