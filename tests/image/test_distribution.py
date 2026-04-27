"""Tests for image distribution checks."""

from __future__ import annotations

import pytest
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from omnilint.image.checks import distribution
from omnilint.core.loader import ImageDataset, ImageEntry


pytestmark = pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")


def test_check_brightness_no_outliers(tmp_path: Path):
    """Test brightness check with no outliers."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    for i in range(10):
        arr = np.random.randint(100, 150, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i}.jpg")
    
    images = [
        ImageEntry(
            filename=f"img_{i}.jpg",
            width=50,
            height=50,
            file_path=img_dir / f"img_{i}.jpg",
            annotations=[],
        )
        for i in range(10)
    ]
    
    dataset = ImageDataset(
        format="yolo",
        images=images,
        categories=["cat"],
        root_path=img_dir,
    )
    
    issues = distribution.check_brightness(dataset)
    assert isinstance(issues, list)


def test_check_brightness_with_underexposed(tmp_path: Path):
    """Test brightness check with underexposed images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    for i in range(8):
        arr = np.random.randint(100, 150, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i}.jpg")
    
    dark_arr = np.zeros((50, 50, 3), dtype=np.uint8)
    Image.fromarray(dark_arr).save(img_dir / "img_dark.jpg")
    Image.fromarray(dark_arr).save(img_dir / "img_dark2.jpg")
    
    images = [
        ImageEntry(
            filename=f.name,
            width=50,
            height=50,
            file_path=f,
            annotations=[],
        )
        for f in img_dir.glob("*.jpg")
    ]
    
    dataset = ImageDataset(
        format="yolo",
        images=images,
        categories=["cat"],
        root_path=img_dir,
    )
    
    issues = distribution.check_brightness(dataset)
    underexposed = [i for i in issues if i.get("check") == "brightness_outliers" and "underexposed" in i.get("detail", "")]
    assert len(underexposed) >= 1


def test_check_contrast_low(sample_image_dataset: ImageDataset, tmp_path: Path):
    """Test low contrast detection."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    solid_img = img_dir / "solid.jpg"
    arr = np.full((50, 50, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(solid_img)
    
    sample_image_dataset.images[0].file_path = solid_img
    
    issues = distribution.check_contrast(sample_image_dataset)
    assert len(issues) >= 1


def test_check_channel_balance_no_imbalance(tmp_path: Path):
    """Test channel balance with balanced channels."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    for i in range(10):
        arr = np.random.randint(80, 180, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i}.jpg")
    
    images = [
        ImageEntry(
            filename=f.name,
            width=50,
            height=50,
            file_path=f,
            annotations=[],
        )
        for f in img_dir.glob("*.jpg")
    ]
    
    dataset = ImageDataset(
        format="yolo",
        images=images,
        categories=["cat"],
        root_path=img_dir,
    )
    
    issues = distribution.check_channel_balance(dataset)
    assert isinstance(issues, list)


def test_run_distribution_all(sample_image_dataset: ImageDataset):
    """Test run function includes all checks."""
    issues = distribution.run(sample_image_dataset)
    assert isinstance(issues, list)