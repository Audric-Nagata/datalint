"""Image-specific pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from dataclasses import dataclass, field

import pytest
from auditor.core.loader import ImageDataset, ImageEntry


@pytest.fixture
def sample_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    return img_dir


@pytest.fixture
def sample_yolo_dir(tmp_path: Path) -> Path:
    """Create a YOLO-style directory structure."""
    root = tmp_path / "yolo_dataset"
    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"
    train_img.mkdir(parents=True)
    train_lbl.mkdir(parents=True)
    
    (root / "data.yaml").write_text("train: train/images\nval: val/images\nnc: 2\nnames: ['cat', 'dog']")
    
    return root


@pytest.fixture
def sample_coco_file(tmp_path: Path) -> Path:
    """Create a sample COCO annotation file."""
    coco = {
        "images": [
            {"id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img_002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 50, 50]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [100, 100, 80, 80]},
        ],
        "categories": [
            {"id": 0, "name": "cat"},
            {"id": 1, "name": "dog"},
        ],
    }
    import json
    coco_file = tmp_path / "annotations.json"
    coco_file.write_text(json.dumps(coco))
    return coco_file


@pytest.fixture
def sample_image_dataset(tmp_path: Path) -> ImageDataset:
    """Create a sample ImageDataset for testing."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    images = []
    for i in range(5):
        img_path = img_dir / f"img_{i:03d}.jpg"
        images.append(ImageEntry(
            filename=img_path.name,
            width=640,
            height=480,
            file_path=img_path,
            annotations=[],
            split="train",
        ))
    
    return ImageDataset(
        format="yolo",
        images=images,
        categories=["cat", "dog"],
        split_col=None,
        root_path=img_dir,
    )


@pytest.fixture
def synthetic_image_file(tmp_path: Path) -> Path:
    """Create a synthetic test image file."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        pytest.skip("PIL not installed")
    
    img_path = tmp_path / "test_image.jpg"
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path