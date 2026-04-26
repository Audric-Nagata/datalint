"""CLI entry point using Typer."""

import sys
from pathlib import Path

import typer
from rich import print as rprint

from auditor import AuditEngine
from auditor.core import loader
from auditor.core.engine import AuditConfig
from auditor.report import builder, renderer_json, renderer_html, renderer_cli

app = typer.Typer(help="DataLint - AI Dataset Quality Auditor")


@app.command()
def run(
    source: str = typer.Argument(..., help="Path to dataset file (CSV or Parquet)"),
    target: str = typer.Option(None, help="Target/label column name"),
    output: str = typer.Option(None, "--output", help="Output file path (JSON or HTML)"),
    format: str = typer.Option("auto", "--format", help="Output format: auto, json, html"),
    fail_below: int = typer.Option(None, "--fail-below", help="Exit code 1 if DQS < threshold"),
    checks: str = typer.Option("all", "--checks", help="Comma-separated list of checks to run"),
    missing_threshold: float = typer.Option(0.05, "--missing-threshold", help="Missing value threshold"),
    dedup_threshold: float = typer.Option(0.95, "--dedup-threshold", help="Near-duplicate threshold"),
) -> None:
    """Run data quality audit on a dataset."""
    try:
        df, schema = loader.load(source)
    except Exception as e:
        rprint(f"[red]Error loading data:[/] {e}")
        raise typer.Exit(1)

    check_list = ["basic", "distribution", "labels", "leakage", "importance", "dedup"]
    if checks != "all":
        check_list = [c.strip() for c in checks.split(",")]

    config = AuditConfig(
        target_col=target,
        checks=check_list,
        missing_threshold=missing_threshold,
        dedup_threshold=dedup_threshold,
        fail_below=fail_below,
    )

    engine = AuditEngine(df, config)
    result = engine.run()

    report = builder.build(result, schema.source, schema.rows, schema.columns)

    if output:
        ext = Path(output).suffix.lower()
        if ext == ".json":
            renderer_json.render(report, output)
            rprint(f"[green]Report saved to {output}[/]")
        elif ext == ".html":
            renderer_html.render(report, output)
            rprint(f"[green]Report saved to {output}[/]")
        else:
            if format == "json":
                renderer_json.render(report, output)
                rprint(f"[green]Report saved to {output}[/]")
            elif format == "html":
                renderer_html.render(report, output)
                rprint(f"[green]Report saved to {output}[/]")
            else:
                renderer_json.render(report, output.replace(".txt", ".json"))
                rprint(f"[green]Report saved to {output}[/]")
    else:
        renderer_cli.render(report)

    if fail_below and result.quality_score < fail_below:
        rprint(f"\n[red]DQS ({result.quality_score}) is below threshold ({fail_below})[/]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from auditor import __version__
    rprint(f"DataLint version: {__version__}")


if __name__ == "__main__":
    app()