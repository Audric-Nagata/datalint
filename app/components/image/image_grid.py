"""Image grid component for visualizing flagged images."""

import streamlit as st
from pathlib import Path


def render(report, image_dataset=None):
    """Render a grid of flagged images with severity badges.
    
    Args:
        report: AuditReport object
        image_dataset: ImageDataset object (optional)
    """
    if image_dataset is None:
        image_dataset = st.session_state.get("image_dataset")
    
    if image_dataset is None:
        st.info("No image dataset loaded")
        return
    
    flagged_by_check = {}
    for issue in report.issues:
        if issue.asset:
            check = issue.check
            if check not in flagged_by_check:
                flagged_by_check[check] = []
            if isinstance(issue.asset, list):
                flagged_by_check[check].extend(issue.asset)
            else:
                flagged_by_check[check].append(issue.asset)
    
    if not flagged_by_check:
        st.info("No flagged images found")
        return
    
    st.markdown("### Filter by Check")
    check_filter = st.selectbox(
        "Select check type",
        options=["All"] + list(flagged_by_check.keys()),
    )
    
    severity_filter = st.selectbox(
        "Select severity",
        options=["All", "critical", "high", "medium", "low"],
    )
    
    if check_filter == "All":
        selected_check = None
    else:
        selected_check = check_filter
    
    filtered_issues = []
    for issue in report.issues:
        if severity_filter != "All" and issue.severity != severity_filter:
            continue
        if selected_check and issue.check != selected_check:
            continue
        if issue.asset:
            filtered_issues.append(issue)
    
    if not filtered_issues:
        st.info("No images match the filter")
        return
    
    st.markdown(f"#### {len(filtered_issues)} Flagged Images")
    
    cols = st.columns(4)
    for idx, issue in enumerate(filtered_issues):
        col = cols[idx % 4]
        
        severity_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }.get(issue.severity, "⚪")
        
        with col:
            st.markdown(f"**{severity_emoji} {issue.check}**")
            st.markdown(f"_{issue.detail}_")
            
            if issue.suggestion:
                with st.expander("Suggestion"):
                    st.write(issue.suggestion)
            
            st.divider()


def render_thumbnail(image_path: Path, caption: str = "", width: int = 150):
    """Render a single thumbnail image.
    
    Args:
        image_path: Path to image file
        caption: Caption text
        width: Display width in pixels
    """
    try:
        st.image(
            str(image_path),
            caption=caption,
            width=width,
        )
    except Exception as e:
        st.warning(f"Could not load image: {e}")