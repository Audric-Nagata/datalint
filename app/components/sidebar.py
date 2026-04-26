"""Streamlit UI sidebar with config controls."""

import streamlit as st
from auditor.core.engine import AuditConfig


def render() -> AuditConfig:
    """Render sidebar and return validated config."""
    st.sidebar.title("DataLint")

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    target_col = st.sidebar.text_input("Target column (optional)", "")
    split_col = st.sidebar.text_input("Split column (optional)", "")

    st.sidebar.markdown("### Checks")
    all_checks = st.sidebar.checkbox("Run all checks", value=True, key="run_all")
    if all_checks:
        check_list = ["basic", "distribution", "labels", "leakage", "importance", "dedup"]
    else:
        check_list = st.sidebar.multiselect(
            "Select checks",
            ["basic", "distribution", "labels", "leakage", "importance", "dedup"],
            default=["basic", "distribution", "labels", "leakage", "importance", "dedup"],
        )

    st.sidebar.markdown("### Thresholds")
    missing_thresh = st.sidebar.slider("Missing value threshold", 0.0, 1.0, 0.05, 0.01)
    dedup_thresh = st.sidebar.slider("Near-duplicate threshold", 0.5, 1.0, 0.95, 0.01)

    st.sidebar.markdown("### Options")
    fail_below = st.sidebar.number_input("Fail below DQS", min_value=0, max_value=100, value=70)

    run_button = st.sidebar.button("Run Audit", type="primary")

    config = AuditConfig(
        target_col=target_col or None,
        split_col=split_col or None,
        checks=check_list,
        missing_threshold=missing_thresh,
        dedup_threshold=dedup_thresh,
        fail_below=fail_below,
    )

    return uploaded_file, config, run_button