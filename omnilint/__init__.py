"""DataLint - AI Dataset Quality omnilint.

Supports tabular (CSV/Parquet) and image (COCO/YOLO) datasets.
"""

from omnilint.core.engine import AuditEngine, AuditConfig, IssueRecord, AuditResult
from omnilint.core.loader import load, ImageDataset, ImageEntry

__version__ = "0.2.0"

__all__ = [
    "AuditEngine",
    "AuditConfig",
    "IssueRecord",
    "AuditResult",
    "load",
    "ImageDataset",
    "ImageEntry",
]