"""CLI entry point using Typer."""

import sys
from pathlib import Path

import typer
from rich import print as rprint

from omnilint import AuditEngine
from omnilint.core import loader
from omnilint.core.engine import AuditConfig
from omnilint.report import builder as report_builder
from omnilint.report import renderer_json, renderer_html, renderer_cli

app = typer.Typer(help="OmniLint - AI Dataset Quality auditor")


@app.command()
def run(
    source: str = typer.Argument(..., help="Path to dataset file (CSV/Parquet) or directory (COCO/YOLO)"),
    target: str = typer.Option(None, help="Target/label column name (tabular mode)"),
    mode: str = typer.Option("tabular", "--mode", help="Audit mode: tabular or image"),
    format: str = typer.Option("auto", "--format", help="Input format: csv, parquet, coco, yolo, auto"),
    output: str = typer.Option(None, "--output", help="Output file path (JSON or HTML)"),
    fail_below: int = typer.Option(None, "--fail-below", help="Exit code 1 if DQS < threshold"),
    checks: str = typer.Option("all", "--checks", help="Comma-separated list of checks to run"),
    missing_threshold: float = typer.Option(0.05, "--missing-threshold", help="Missing value threshold"),
    dedup_threshold: float = typer.Option(0.95, "--dedup-threshold", help="Near-duplicate threshold"),
    blur_threshold: float = typer.Option(100, "--blur-threshold", help="Blur detection threshold (image mode)"),
) -> None:
    """Run data quality audit on a dataset."""
    try:
        data, schema = loader.load(source)
    except Exception as e:
        rprint(f"[red]Error loading data:[/] {e}")
        raise typer.Exit(1)

    if mode == "image":
        _run_image_audit(data, schema, checks, blur_threshold, output, fail_below)
    else:
        _run_tabular_audit(data, schema, target, checks, missing_threshold, dedup_threshold, output, fail_below)


def _run_tabular_audit(
    df,
    schema,
    target,
    checks,
    missing_threshold,
    dedup_threshold,
    output,
    fail_below,
):
    """Run tabular audit."""
    check_list = ["basic", "distribution", "labels", "leakage", "importance", "dedup"]
    if checks != "all":
        check_list = [c.strip() for c in checks.split(",")]

    config = AuditConfig(
        target_col=target,
        checks=check_list,
        missing_threshold=missing_threshold,
        dedup_threshold=dedup_threshold,
        fail_below=fail_below,
        mode="tabular",
    )

    engine = AuditEngine(df, config)
    result = engine.run()

    report = report_builder.build(result, schema.source, schema.rows, schema.columns, mode="tabular")

    if output:
        _save_report(report, output)
    else:
        renderer_cli.render(report)

    if fail_below and result.quality_score < fail_below:
        rprint(f"\n[red]DQS ({result.quality_score}) is below threshold ({fail_below})[/]")
        raise typer.Exit(1)


def _run_image_audit(
    dataset,
    schema,
    checks,
    blur_threshold,
    output,
    fail_below,
):
    """Run image audit."""
    check_list = ["integrity", "distribution", "labels", "duplicates", "anomalies"]
    if checks != "all":
        check_list = [c.strip() for c in checks.split(",")]

    config = AuditConfig(
        checks=check_list,
        missing_threshold=blur_threshold,
        fail_below=fail_below,
        mode="image",
    )

    engine = AuditEngine(dataset, config)
    result = engine.run()

    report = report_builder.build(
        result,
        source=str(schema),
        rows=len(dataset.images),
        columns=len(dataset.categories),
        mode="image",
    )

    if output:
        _save_report(report, output)
    else:
        renderer_cli.render(report)

    if fail_below and result.quality_score < fail_below:
        rprint(f"\n[red]DQS ({result.quality_score}) is below threshold ({fail_below})[/]")
        raise typer.Exit(1)


def _save_report(report, output):
    """Save report to file."""
    ext = Path(output).suffix.lower()
    if ext == ".json":
        renderer_json.render(report, output)
        rprint(f"[green]Report saved to {output}[/]")
    elif ext == ".html":
        renderer_html.render(report, output)
        rprint(f"[green]Report saved to {output}[/]")
    else:
        renderer_json.render(report, output.replace(".txt", ".json"))
        rprint(f"[green]Report saved to {output}[/]")


@app.command()
def version() -> None:
    """Show version information."""
    from omnilint import __version__
    rprint(f"OmniLint version: {__version__}")


if __name__ == "__main__":
    app()