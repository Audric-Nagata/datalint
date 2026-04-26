"""CLI report renderer using Rich."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from auditor.report.builder import AuditReport


def render(report: AuditReport) -> None:
    """Render AuditReport to terminal."""
    console = Console()

    score_color = get_score_color(report.score_band)

    console.print(Panel.fit(
        f"[bold {score_color}]{report.quality_score}[/] / 100  —  {report.score_band}",
        title="Data Quality Score",
        border_style=score_color,
    ))

    console.print()
    console.print(f"[bold]File:[/] {report.metadata['filename']}")
    console.print(f"[bold]Rows:[/] {report.metadata['rows']}, [bold]Cols:[/] {report.metadata['columns']}")

    console.print()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Severity", style="bold")
    table.add_column("Check")
    table.add_column("Column")
    table.add_column("Detail")

    for issue in report.issues[:20]:
        severity_style = get_severity_style(issue.severity)
        table.add_row(
            f"[{severity_style}]{issue.severity.upper()}[/{severity_style}]",
            issue.check,
            issue.column or "-",
            issue.detail[:60] + "..." if len(issue.detail) > 60 else issue.detail,
        )

    console.print(table)

    if report.summary["total"] > 20:
        console.print(f"\n[dim]... and {report.summary['total'] - 20} more issues[/]")

    console.print(f"\n[dim]Run with --output report.html for full report.[/]")


def get_score_color(band: str) -> str:
    """Get color for score band."""
    colors = {"Critical": "red", "Poor": "yellow", "Fair": "green", "Good": "bright_green"}
    return colors.get(band, "white")


def get_severity_style(severity: str) -> str:
    """Get Rich style for severity."""
    styles = {"critical": "bold red", "high": "bold orange1", "medium": "bold yellow", "low": "bold green"}
    return styles.get(severity, "white")