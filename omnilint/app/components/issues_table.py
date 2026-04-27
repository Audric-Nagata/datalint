"""Issues table component: filterable issues breakdown."""

import streamlit as st
from omnilint.report.builder import AuditReport


def render(report: AuditReport) -> None:
    """Render issues table."""
    if not report.issues:
        st.info("No issues found. Dataset looks clean!")
        return

    severity_filter = st.multiselect(
        "Filter by severity",
        ["critical", "high", "medium", "low"],
        default=["critical", "high", "medium", "low"],
    )

    module_filter = st.multiselect(
        "Filter by module",
        list(set(i.check for i in report.issues)),
        default=list(set(i.check for i in report.issues)),
    )

    filtered = [
        i
        for i in report.issues
        if i.severity in severity_filter and i.check in module_filter
    ]

    severity_colors = {
        "critical": "red",
        "high": "orange",
        "medium": "yellow",
        "low": "green",
    }

    data = []
    for i in filtered:
        data.append(
            {
                "Severity": i.severity.upper(),
                "Check": i.check,
                "Column": i.column or "-",
                "Detail": i.detail,
                "Suggestion": i.suggestion,
            }
        )

    if data:
        st.dataframe(
            data,
            column_config={
                "Severity": st.column_config.TextColumn(
                    "Severity",
                    help="Issue severity",
                )
            },
        )

    for issue in filtered:
        with st.expander(f"{issue.check}: {issue.column or '-'} ({issue.severity})"):
            st.markdown(f"**Detail:** {issue.detail}")
            st.markdown(f"**Suggestion:** {issue.suggestion}")