"""Score card component: DQS gauge and band display."""

import streamlit as st
from auditor.report.builder import AuditReport


def render(report: AuditReport) -> None:
    """Render DQS score card."""
    score = report.quality_score
    band = report.score_band

    score_color = {
        "Critical": "#dc2626",
        "Poor": "#ea580c",
        "Fair": "#ca8a04",
        "Good": "#16a34a",
    }.get(band, "#666")

    st.markdown(
        f"""
        <div style="text-align: center; padding: 24px; background: white; border-radius: 12px; margin-bottom: 24px;">
            <div style="font-size: 72px; font-weight: bold; color: {score_color};">{score}</div>
            <div style="font-size: 24px; font-weight: 600; color: {score_color};>{band}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", report.metadata["rows"])
    with col2:
        st.metric("Columns", report.metadata["columns"])
    with col3:
        st.metric("Issues", report.summary["total"])
    with col4:
        st.metric("Critical/High", report.summary["critical"] + report.summary["high"])

    if report.module_scores:
        st.markdown("### Module Scores")
        for module, score_val in report.module_scores.items():
            st.progress(score_val / 100, text=f"{module}: {score_val}")