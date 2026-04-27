"""Image module explorer for detailed per-module analysis."""

import streamlit as st


def render(report):
    """Render image module explorer panels.
    
    Args:
        report: AuditReport object
    """
    st.markdown("### Image Module Explorer")
    
    if not hasattr(report, "module_scores") or not report.module_scores:
        st.info("No module scores available")
        return
    
    integrity_exp = st.expander("📷 Integrity Checks", expanded=True)
    with integrity_exp:
        _render_module_issues(report, "integrity")
    
    dist_exp = st.expander("📊 Distribution Analysis")
    with dist_exp:
        _render_module_issues(report, "distribution")
    
    labels_exp = st.expander(" 🏷️ Label Checks")
    with labels_exp:
        _render_module_issues(report, "labels")
    
    dup_exp = st.expander("🔄 Duplicate Checks")
    with dup_exp:
        _render_module_issues(report, "duplicates")
    
    anomaly_exp = st.expander("⚠️ Anomaly Checks")
    with anomaly_exp:
        _render_module_issues(report, "anomalies")


def _render_module_issues(report, module_name: str):
    """Render issues for a specific module."""
    module_issues = [
        i for i in report.issues if i.check.startswith(module_name) or _check_module(i.check, module_name)
    ]
    
    if not module_issues:
        st.info(f"No {module_name} issues found")
        return
    
    score = report.module_scores.get(module_name, 100.0)
    st.progress(score / 100.0, text=f"Score: {score:.1f}")
    
    for issue in module_issues:
        severity_badge = _severity_badge(issue.severity)
        st.markdown(f"**{severity_badge} {issue.check}**")
        st.markdown(f"_{issue.detail}_")
        if issue.suggestion:
            st.caption(f"💡 {issue.suggestion}")
        if issue.asset:
            st.caption(f"📁 Assets: {issue.asset}")
        st.divider()


def _check_module(check_name: str, module: str) -> bool:
    """Check if a check belongs to a module."""
    module_checks = {
        "integrity": ["corrupt", "resolution", "format"],
        "distribution": ["brightness", "contrast", "channel", "exposure"],
        "labels": ["label_file", "class_imbalance", "missing"],
        "duplicates": ["duplicate", "near_dup"],
        "anomalies": ["blur", "blank", "mislabel"],
    }
    keywords = module_checks.get(module, [])
    return any(kw in check_name.lower() for kw in keywords)


def _severity_badge(severity: str) -> str:
    """Get severity badge emoji."""
    badges = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
    }
    return badges.get(severity, "⚪")


def render_resolution_scatter(image_dataset):
    """Render resolution scatter plot."""
    if not image_dataset:
        st.info("No image dataset loaded")
        return
    
    widths = [img.width for img in image_dataset.images]
    heights = [img.height for img in image_dataset.images]
    
    if not widths:
        st.info("No image dimensions available")
        return
    
    import numpy as np
    median_w = np.median(widths)
    median_h = np.median(heights)
    
    st.markdown(f"Median resolution: {median_w:.0f} x {median_h:.0f}")
    
    data = {
        "x": widths,
        "y": heights,
        "mode": [img.filename for img in image_dataset.images],
    }
    
    try:
        import plotly.express as px
        fig = px.scatter(
            data,
            x="x",
            y="y",
            hover_name="mode",
            title="Image Resolution Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly for visualization")


def render_brightness_histogram(image_dataset):
    """Render brightness histogram."""
    if not image_dataset:
        st.info("No image dataset loaded")
        return
    
    st.info("Brightness histogram - calculate pixel stats first")


def render_duplicate_clusters(report):
    """Render duplicate cluster viewer."""
    dup_issues = [i for i in report.issues if "duplicate" in i.check.lower()]
    
    if not dup_issues:
        st.info("No duplicate clusters found")
        return
    
    for issue in dup_issues:
        st.markdown(f"**{issue.check}**: {issue.detail}")
        if issue.suggestion:
            st.caption(issue.suggestion)