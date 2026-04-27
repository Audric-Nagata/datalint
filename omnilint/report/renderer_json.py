"""JSON report renderer."""

import json
from omnilint.report.builder import AuditReport


def render(report: AuditReport, output_path: str) -> None:
    """Render AuditReport to JSON file."""
    data = {
        "metadata": report.metadata,
        "quality_score": report.quality_score,
        "score_band": report.score_band,
        "issues": [
            {
                "check": i.check,
                "column": i.column,
                "severity": i.severity,
                "detail": i.detail,
                "suggestion": i.suggestion,
            }
            for i in report.issues
        ],
        "summary": report.summary,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)