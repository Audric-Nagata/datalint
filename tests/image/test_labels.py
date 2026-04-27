"""Tests for image label checks."""

from __future__ import annotations

import pytest
from pathlib import Path

from auditor.image.checks import labels
from auditor.core.loader import ImageDataset, ImageEntry


def test_check_label_file_mismatch_none(sample_image_dataset: ImageDataset):
    """Test when manifest matches disk files."""
    for img in sample_image_dataset.images:
        img.file_path.touch()
        if img.file_path.parent.name != "images":
            img.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    sample_image_dataset.root_path = sample_image_dataset.images[0].file_path.parent
    
    issues = labels.check_label_file_mismatch(sample_image_dataset)
    assert len(issues) == 0


def test_check_label_file_mismatch_exists(sample_image_dataset: ImageDataset):
    """Test when there's a mismatch."""
    sample_image_dataset.root_path = Path("/fake/path")
    
    issues = labels.check_label_file_mismatch(sample_image_dataset)
    assert len(issues) >= 1
    assert issues[0]["check"] == "label_file_mismatch"


def test_check_class_imbalance_no_imbalance(sample_image_dataset: ImageDataset):
    """Test with balanced classes."""
    for img in sample_image_dataset.images[:3]:
        img.annotations = [{"category_id": 0}]
    for img in sample_image_dataset.images[3:]:
        img.annotations = [{"category_id": 1}]
    
    issues = labels.check_class_imbalance(sample_image_dataset)
    assert len(issues) == 0


def test_check_class_imbalance_with_imbalance(sample_image_dataset: ImageDataset):
    """Test with imbalanced classes."""
    for img in sample_image_dataset.images[:9]:
        img.annotations = [{"category_id": 0}]
    for img in sample_image_dataset.images[9:]:
        img.annotations = [{"category_id": 1}]
    
    sample_image_dataset.categories = ["cat", "dog"]
    
    issues = labels.check_class_imbalance(sample_image_dataset)
    assert len(issues) >= 1


def test_check_class_imbalance_no_categories(sample_image_dataset: ImageDataset):
    """Test when no categories defined."""
    sample_image_dataset.categories = []
    
    issues = labels.check_class_imbalance(sample_image_dataset)
    assert len(issues) == 0


def test_run_labels_all(sample_image_dataset: ImageDataset):
    """Test run function includes all checks."""
    issues = labels.run(sample_image_dataset)
    assert isinstance(issues, list)