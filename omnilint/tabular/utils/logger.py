"""Structured logging utilities."""

import logging
import sys
from typing import Any


def get_logger(name: str = "omnilint") -> logging.Logger:
    """Get configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_issue(logger: logging.Logger, issue: dict) -> None:
    """Log an issue at appropriate level."""
    severity = issue.get("severity", "low")
    level = {
        "critical": logger.critical,
        "high": logger.error,
        "medium": logger.warning,
        "low": logger.info,
    }.get(severity, logger.info)

    level(f"{issue['check']}: {issue['detail']}")