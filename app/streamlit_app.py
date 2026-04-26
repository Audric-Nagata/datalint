"""Main Streamlit UI entrypoint."""

import streamlit as st
import pandas as pd
from io import StringIO

from auditor import AuditEngine
from auditor.core import loader
from auditor.core.engine import AuditConfig
from auditor.report import builder
from app.components import sidebar, score_card, issues_table, distribution_charts, module_explorer

st.set_page_config(page_title="DataLint", page_icon="", layout="wide")

st.title("DataLint - Dataset Quality Auditor")

uploaded_file, config, run_button = sidebar.render()

if uploaded_file and run_button:
    try:
        bytes_data = uploaded_file.read()
        df = pd.read_csv(StringIO(bytes_data.decode("utf-8")))

        engine = AuditEngine(df, config)
        result = engine.run()

        schema_info = loader.infer_schema(df)
        report = builder.build(
            result,
            source=uploaded_file.name,
            rows=len(df),
            columns=len(df.columns),
        )

        st.session_state["report"] = report
        st.session_state["df"] = df

    except Exception as e:
        st.error(f"Error running audit: {e}")

if "report" in st.session_state:
    report = st.session_state["report"]
    score_card.render(report)

    tab1, tab2, tab3 = st.tabs(["Issues", "Distributions", "Modules"])

    with tab1:
        issues_table.render(report)

    with tab2:
        if "df" in st.session_state:
            distribution_charts.render(st.session_state["df"])

    with tab3:
        module_explorer.render(report)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download JSON",
            data=str(report.metadata),
            file_name="report.json",
            mime="application/json",
        )