"""HTML report renderer using Jinja2."""

from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from omnilint.report.builder import AuditReport


def render(report: AuditReport, output_path: str, template_path: str | None = None) -> None:
    """Render AuditReport to HTML file."""
    if template_path is None:
        template_dir = Path(__file__).parent.parent / "templates"
        if template_dir.exists():
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template("report.html.j2")
        else:
            template = get_default_template()
    else:
        env = Environment(loader=FileSystemLoader(Path(template_path).parent))
        template = env.get_template(Path(template_path).name)

    context = build_context(report)
    html = template.render(**context)

    with open(output_path, "w") as f:
        f.write(html)


def build_context(report: AuditReport) -> dict:
    """Build template context from AuditReport."""
    severity_colors = {
        "critical": "#dc2626",
        "high": "#ea580c",
        "medium": "#ca8a04",
        "low": "#16a34a",
    }

    return {
        "metadata": report.metadata,
        "quality_score": report.quality_score,
        "score_band": report.score_band,
        "issues": report.issues,
        "summary": report.summary,
        "module_scores": report.module_scores,
        "severity_colors": severity_colors,
    }


def get_default_template() -> "Template":
    """Return a simple default template if no template file exists."""

    class SimpleTemplate:
        def render(self, **context) -> str:
            issues_html = ""
            for issue in context.get("issues", []):
                color = context["severity_colors"].get(issue.severity, "#666")
                issues_html += f'''
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px; color: {color}; font-weight: bold;">{issue.severity.upper()}</td>
                    <td style="padding: 8px;">{issue.check}</td>
                    <td style="padding: 8px;">{issue.column or "-"}</td>
                    <td style="padding: 8px;">{issue.detail}</td>
                </tr>
                '''

            return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Data Quality Audit Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; }}
        .header {{ margin-bottom: 30px; }}
        .score {{ font-size: 48px; font-weight: bold; }}
        .score-critical {{ color: #dc2626; }}
        .score-poor {{ color: #ea580c; }}
        .score-fair {{ color: #ca8a04; }}
        .score-good {{ color: #16a34a; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; padding: 12px; background: #f9fafb; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Data Quality Audit Report</h1>
        <p>File: {context["metadata"]["filename"]}</p>
        <p>Rows: {context["metadata"]["rows"]}, Columns: {context["metadata"]["columns"]}</p>
    </div>
    <div class="score">
        <span class="score-{context["score_band"].lower()}">{context["quality_score"]}</span>
        <span>/ 100 — {context["score_band"]}</span>
    </div>
    <h2>Issues ({context["summary"]["total"]})</h2>
    <table>
        <thead>
            <tr>
                <th>Severity</th>
                <th>Check</th>
                <th>Column</th>
                <th>Detail</th>
            </tr>
        </thead>
        <tbody>{issues_html}</tbody>
    </table>
</body>
</html>'''
    return SimpleTemplate()