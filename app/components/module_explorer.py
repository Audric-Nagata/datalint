"""Module explorer: per-module expandable detail panels."""

import streamlit as st
from auditor.report.builder import AuditReport


def render(report: AuditReport, module_details: dict | None = None) -> None:
    """Render module explorer panels."""
    module_details = module_details or {}

    for module_name in ["basic", "distribution", "labels", "leakage", "importance", "dedup"]:
        module_issues = [i for i in report.issues if i.check.startswith(module_name)]

        with st.expander(f"{module_name.capitalize()} Details"):
            if module_issues:
                for issue in module_issues:
                    st.markdown(f"**{issue.check}**")
                    st.markdown(f"_{issue.severity}_: {issue.detail}")
                    st.markdown(f"> {issue.suggestion}")
            else:
                st.info(f"No {module_name} issues found")

            if module_name in module_details:
                st.json(module_details[module_name])