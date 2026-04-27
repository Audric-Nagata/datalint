"""Main audit engine orchestrator."""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Literal

from auditor.core import loader, scorer

from auditor.tabular.checks import (
    basic,
    distribution,
    labels,
    leakage,
    importance,
    dedup,
)
import auditor.image.checks as image_checks

from auditor.core.loader import ImageDataset


@dataclass
class AuditConfig:
    target_col: str | None = None
    split_col: str | None = None
    checks: list[str] = field(
        default_factory=lambda: [
            "basic",
            "distribution",
            "labels",
            "leakage",
            "importance",
            "dedup",
        ]
    )
    missing_threshold: float = 0.05
    dedup_threshold: float = 0.95
    fail_below: int | None = None
    mode: Literal["tabular", "image"] = "tabular"


@dataclass
class IssueRecord:
    check: str
    severity: str
    detail: str
    suggestion: str
    column: str | None = None
    asset: str | None = None


@dataclass
class AuditResult:
    quality_score: float
    score_band: str
    total_issues: int
    issues: list[IssueRecord]
    module_scores: dict[str, float] = field(default_factory=dict)


TABULAR_CHECKS = ["basic", "distribution", "labels", "leakage", "importance", "dedup"]
IMAGE_CHECKS = ["integrity", "distribution", "labels", "duplicates", "anomalies"]


class AuditEngine:
    def __init__(
        self,
        data: pd.DataFrame | ImageDataset,
        config: AuditConfig | None = None,
    ):
        self.data = data
        self.config = config or AuditConfig()
        self._mode = self._detect_mode()

    def _detect_mode(self) -> Literal["tabular", "image"]:
        if isinstance(self.data, ImageDataset):
            return "image"
        return "tabular"

    def run(self) -> AuditResult:
        """Execute the full audit pipeline."""
        module_findings = {}
        self._validate_config()

        if self._mode == "tabular":
            return self._run_tabular()
        else:
            return self._run_image()

    def _run_tabular(self) -> AuditResult:
        """Run tabular audit pipeline."""
        module_findings = {}
        df = self.data

        for check_name in self.config.checks:
            if check_name == "basic":
                module_findings["basic"] = self._run_basic()
            elif check_name == "distribution":
                module_findings["distribution"] = self._run_distribution()
            elif check_name == "labels":
                module_findings["labels"] = self._run_labels()
            elif check_name == "leakage":
                module_findings["leakage"] = self._run_leakage()
            elif check_name == "importance":
                module_findings["importance"] = self._run_importance()
            elif check_name == "dedup":
                module_findings["dedup"] = self._run_dedup()

        score_result = scorer.compute(module_findings, mode="tabular")
        issues = self._flatten_issues(module_findings)

        module_scores = {}
        for module, findings in module_findings.items():
            if findings:
                mod_score = scorer.compute({module: findings}, mode="tabular")
                module_scores[module] = mod_score.quality_score
            else:
                module_scores[module] = 100.0

        return AuditResult(
            quality_score=score_result.quality_score,
            score_band=score_result.score_band,
            total_issues=len(issues),
            issues=issues,
            module_scores=module_scores,
        )

    def _run_image(self) -> AuditResult:
        """Run image audit pipeline."""
        module_findings = {}
        dataset = self.data

        for check_name in self.config.checks:
            if check_name == "integrity":
                module_findings["integrity"] = self._run_integrity()
            elif check_name == "distribution":
                module_findings["distribution"] = self._run_image_distribution()
            elif check_name == "labels":
                module_findings["labels"] = self._run_image_labels()
            elif check_name == "duplicates":
                module_findings["duplicates"] = self._run_duplicates()
            elif check_name == "anomalies":
                module_findings["anomalies"] = self._run_anomalies()

        score_result = scorer.compute(module_findings, mode="image")
        issues = self._flatten_issues(module_findings)

        module_scores = {}
        for module, findings in module_findings.items():
            if findings:
                mod_score = scorer.compute({module: findings}, mode="image")
                module_scores[module] = mod_score.quality_score
            else:
                module_scores[module] = 100.0

        return AuditResult(
            quality_score=score_result.quality_score,
            score_band=score_result.score_band,
            total_issues=len(issues),
            issues=issues,
            module_scores=module_scores,
        )

    def _validate_config(self) -> None:
        if self._mode == "tabular":
            df = self.data
            if self.config.target_col and self.config.target_col not in df.columns:
                raise ValueError(f"Target column '{self.config.target_col}' not found")

    def _run_basic(self) -> list[dict]:
        """Run basic data checks."""
        return basic.run(self.data, threshold=self.config.missing_threshold)

    def _run_distribution(self) -> list[dict]:
        """Run distribution analysis."""
        return distribution.run(self.data)

    def _run_labels(self) -> list[dict]:
        """Run label checks."""
        if not self.config.target_col:
            return []
        return labels.run(self.data, self.config.target_col)

    def _run_leakage(self) -> list[dict]:
        """Run leakage detection."""
        if not self.config.target_col:
            return []
        return leakage.run(self.data, self.config.target_col)

    def _run_importance(self) -> list[dict]:
        """Run feature importance noise detection."""
        if not self.config.target_col:
            return []
        return importance.run(self.data, self.config.target_col)

    def _run_dedup(self) -> list[dict]:
        """Run duplicate detection."""
        return dedup.run(self.data, threshold=self.config.dedup_threshold)

    def _run_integrity(self) -> list[dict]:
        """Run image integrity checks."""
        return image_checks.integrity.run(self.data)

    def _run_image_distribution(self) -> list[dict]:
        """Run image distribution checks."""
        return image_checks.distribution.run(self.data)

    def _run_image_labels(self) -> list[dict]:
        """Run image label checks."""
        return image_checks.labels.run(self.data)

    def _run_duplicates(self) -> list[dict]:
        """Run image duplicate checks."""
        return image_checks.duplicates.run(self.data)

    def _run_anomalies(self) -> list[dict]:
        """Run image anomaly checks."""
        return image_checks.anomalies.run(self.data)

    def _flatten_issues(self, module_findings: dict[str, list[dict]]) -> list[IssueRecord]:
        """Flatten all findings into IssueRecords sorted by severity."""
        issues = []
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        for module, findings in module_findings.items():
            for f in findings:
                issues.append(
                    IssueRecord(
                        check=f.get("check", module),
                        column=f.get("column"),
                        asset=f.get("asset"),
                        severity=f.get("severity", "low"),
                        detail=f.get("detail", ""),
                        suggestion=f.get("suggestion", ""),
                    )
                )

        issues.sort(key=lambda x: (severity_order.get(x.severity, 3), x.check))
        return issues