"""Streamlit UI sidebar with config controls."""

import streamlit as st
from auditor.core.engine import AuditConfig


def render(mode: str = "tabular") -> tuple:
    """Render sidebar and return validated config.
    
    Args:
        mode: "tabular" or "image"
    
    Returns:
        (uploaded_file, config, run_button)
    """
    st.sidebar.title("DataLint")

    if mode == "tabular":
        return _render_tabular()
    else:
        return _render_image()


def _render_tabular():
    """Render sidebar for tabular mode."""
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload a CSV file for tabular data auditing"
    )

    target_col = st.sidebar.text_input("Target column (optional)", "")
    split_col = st.sidebar.text_input("Split column (optional)", "")

    st.sidebar.markdown("### Checks")
    all_checks = st.sidebar.checkbox("Run all checks", value=True, key="run_all_tabular")
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
        mode="tabular",
    )

    return uploaded_file, config, run_button


def _render_image():
    """Render sidebar for image mode."""
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image Dataset",
        type=["zip"],
        help="Upload a ZIP file containing images or a COCO/YOLO dataset folder"
    )

    st.sidebar.markdown("### Data Format")
    data_format = st.sidebar.selectbox(
        "Dataset format",
        ["auto", "coco", "yolo"],
        index=0,
        help="Auto-detect or specify COCO/YOLO format"
    )

    st.sidebar.markdown("### Image Checks")
    all_checks = st.sidebar.checkbox("Run all checks", value=True, key="run_all_image")
    if all_checks:
        check_list = ["integrity", "distribution", "labels", "duplicates", "anomalies"]
    else:
        check_list = st.sidebar.multiselect(
            "Select checks",
            ["integrity", "distribution", "labels", "duplicates", "anomalies"],
            default=["integrity", "distribution", "labels", "duplicates", "anomalies"],
        )

    st.sidebar.markdown("### Thresholds")
    blur_thresh = st.sidebar.slider("Blur threshold (Laplacian)", 10, 200, 100, 10)
    dup_thresh = st.sidebar.slider("Duplicate similarity threshold", 0.5, 1.0, 0.97, 0.01)

    st.sidebar.markdown("### Options")
    fail_below = st.sidebar.number_input("Fail below DQS", min_value=0, max_value=100, value=70)

    run_button = st.sidebar.button("Run Audit", type="primary")

    config = AuditConfig(
        target_col=None,
        split_col=None,
        checks=check_list,
        missing_threshold=blur_thresh,
        dedup_threshold=dup_thresh,
        fail_below=fail_below,
        mode="image",
    )

    return uploaded_file, config, run_button