"""DataLint - AI Dataset Quality omnilint.

Supports tabular (CSV/Parquet) and image (COCO/YOLO) datasets.
"""

from omnilint.core.engine import AuditEngine, AuditConfig, IssueRecord, AuditResult
from omnilint.core.loader import load, ImageDataset, ImageEntry

from omnilint.cli.main import app as cli_main

__version__ = "0.1.5"

__all__ = [
    "AuditEngine",
    "AuditConfig",
    "IssueRecord",
    "AuditResult",
    "load",
    "ImageDataset",
    "ImageEntry",
    "cli_main",
]