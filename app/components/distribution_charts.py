"""Distribution charts component: per-column distribution plots."""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from auditor.report.builder import AuditReport


def render(df: pd.DataFrame) -> None:
    """Render distribution charts."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns to display.")
        return

    col_to_plot = st.selectbox("Select column", numeric_cols)

    if col_to_plot:
        fig = px.histogram(df, x=col_to_plot, title=f"Distribution of {col_to_plot}")
        st.plotly_chart(fig, use_container_width=True)

        stats = df[col_to_plot].describe()
        st.write(stats)


def render_all(df: pd.DataFrame) -> None:
    """Render all numeric columns in an expander."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return

    with st.expander("View all distributions"):
        cols = st.columns(3)
        for i, col in enumerate(numeric_cols[:9]):
            with cols[i % 3]:
                fig = px.histogram(df, x=col, title=col)
                st.plotly_chart(fig, use_container_width=True)