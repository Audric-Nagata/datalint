"""Main audit engine orchestrator."""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Any

from auditor.core import loader, scorer
from auditor.checks import basic, distribution, labels, leakage, importance, dedup


@dataclass
class AuditConfig:
    target_col: str | None = None
    split_col: str | None = None
    checks: list[str] = field(default_factory=lambda: ["basic", "distribution", "labels", "leakage", "importance", "dedup"])
    missing_threshold: float = 0.05
    dedup_threshold: float = 0.95
    fail_below: int | None = None


@dataclass
class IssueRecord:
    check: str
    severity: str
    detail: str
    suggestion: str
    column: str | None = None


@dataclass
class AuditResult:
    quality_score: float
    score_band: str
    total_issues: int
    issues: list[IssueRecord]
    module_scores: dict[str, float] = field(default_factory=dict)


class AuditEngine:
    def __init__(self, df: pd.DataFrame, config: AuditConfig | None = None):
        self.df = df
        self.config = config or AuditConfig()

    def run(self) -> AuditResult:
        """Execute the full audit pipeline."""
        module_findings = {}
        self._validate_config()

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

        score_result = scorer.compute(module_findings)
        issues = self._flatten_issues(module_findings)

        module_scores = {}
        for module, findings in module_findings.items():
            if findings:
                mod_score = scorer.compute({module: findings})
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
        if self.config.target_col and self.config.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.config.target_col}' not found")

    def _run_basic(self) -> list[dict]:
        """Run basic data checks."""
        return basic.run(self.df, threshold=self.config.missing_threshold)

    def _run_distribution(self) -> list[dict]:
        """Run distribution analysis."""
        return distribution.run(self.df)

    def _run_labels(self) -> list[dict]:
        """Run label checks."""
        if not self.config.target_col:
            return []
        return labels.run(self.df, self.config.target_col)

    def _run_leakage(self) -> list[dict]:
        """Run leakage detection."""
        if not self.config.target_col:
            return []
        return leakage.run(self.df, self.config.target_col)

    def _run_importance(self) -> list[dict]:
        """Run feature importance noise detection."""
        if not self.config.target_col:
            return []
        return importance.run(self.df, self.config.target_col)

    def _run_dedup(self) -> list[dict]:
        """Run duplicate detection."""
        return dedup.run(self.df, threshold=self.config.dedup_threshold)

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
                        severity=f.get("severity", "low"),
                        detail=f.get("detail", ""),
                        suggestion=f.get("suggestion", ""),
                    )
                )

        issues.sort(key=lambda x: (severity_order.get(x.severity, 3), x.check))
        return issues