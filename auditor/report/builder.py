"""Report builder: assembles final AuditReport object."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from auditor.core.engine import AuditResult


@dataclass
class IssueRecord:
    check: str
    severity: str
    detail: str
    suggestion: str
    column: str | None = None


@dataclass
class AuditReport:
    metadata: dict
    quality_score: float
    score_band: str
    issues: list[IssueRecord]
    summary: dict
    module_scores: dict[str, float] = field(default_factory=dict)


def build(audit_result: AuditResult, source: str, rows: int, columns: int) -> AuditReport:
    """Build final AuditReport from AuditResult."""
    summary = {
        "total": audit_result.total_issues,
        "critical": sum(1 for i in audit_result.issues if i.severity == "critical"),
        "high": sum(1 for i in audit_result.issues if i.severity == "high"),
        "medium": sum(1 for i in audit_result.issues if i.severity == "medium"),
        "low": sum(1 for i in audit_result.issues if i.severity == "low"),
    }

    return AuditReport(
        metadata={
            "filename": source,
            "rows": rows,
            "columns": columns,
            "audited_at": datetime.utcnow().isoformat() + "Z",
        },
        quality_score=audit_result.quality_score,
        score_band=audit_result.score_band,
        issues=audit_result.issues,
        summary=summary,
        module_scores=audit_result.module_scores,
    )


def sort_issues(issues: list[IssueRecord]) -> list[IssueRecord]:
    """Sort issues by severity."""
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(issues, key=lambda x: severity_order.get(x.severity, 3))