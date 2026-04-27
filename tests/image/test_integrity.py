"""Tests for image integrity checks."""

from __future__ import annotations

import pytest
from pathlib import Path

from omnilint.image.checks import integrity
from omnilint.core.loader import ImageDataset, ImageEntry


def test_check_corrupt_files_none_exist(sample_image_dataset: ImageDataset):
    """Test when all files exist and are readable."""
    for img in sample_image_dataset.images:
        img.file_path.touch()
    
    issues = integrity.check_corrupt_files(sample_image_dataset)
    assert len(issues) == 0


def test_check_corrupt_files_with_missing(sample_image_dataset: ImageDataset):
    """Test when some files are missing."""
    for img in sample_image_dataset.images[:2]:
        pass
    
    issues = integrity.check_corrupt_files(sample_image_dataset)
    assert len(issues) == 1
    assert issues[0]["check"] == "corrupt_files"
    assert issues[0]["severity"] == "critical"


def test_check_resolution_outliers(sample_image_dataset: ImageDataset):
    """Test resolution outlier detection."""
    sample_image_dataset.images[0].width = 10
    sample_image_dataset.images[0].height = 10
    
    issues = integrity.check_resolution(sample_image_dataset)
    assert len(issues) >= 1
    assert issues[0]["check"] == "resolution_outlier"


def test_check_format_consistency_mixed(sample_image_dataset: ImageDataset):
    """Test format inconsistency detection."""
    sample_image_dataset.images[0].file_path = Path("test.jpg")
    sample_image_dataset.images[1].file_path = Path("test.png")
    
    issues = integrity.check_format_consistency(sample_image_dataset)
    assert len(issues) == 1
    assert issues[0]["check"] == "format_inconsistency"


def test_check_format_consistency_single(sample_image_dataset: ImageDataset):
    """Test when all formats are consistent."""
    for img in sample_image_dataset.images:
        img.file_path = Path("test.jpg")
    
    issues = integrity.check_format_consistency(sample_image_dataset)
    assert len(issues) == 0


def test_run_integrity_all_checks(sample_image_dataset: ImageDataset):
    """Test run function runs all checks."""
    for img in sample_image_dataset.images:
        img.file_path.touch()
    
    sample_image_dataset.images[0].width = 10
    sample_image_dataset.images[0].height = 10
    
    issues = integrity.run(sample_image_dataset)
    assert len(issues) >= 1