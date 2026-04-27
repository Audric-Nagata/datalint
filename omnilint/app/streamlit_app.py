"""Main Streamlit UI entrypoint."""

from omnilint.app.components import distribution_charts, issues_table, module_explorer, score_card
import streamlit as st
import pandas as pd
import zipfile
import tempfile
from pathlib import Path
from io import StringIO

from omnilint.core import loader
from omnilint.core.engine import AuditEngine, AuditConfig
from omnilint.report import builder as report_builder

from omnilint.app.components import (
    sidebar,
)
from omnilint.app.components.image import image_module_explorer

st.set_page_config(page_title="OmniLint", page_icon="", layout="wide")

st.title("OmniLint - Dataset Quality auditor")

mode = st.radio(
    "Select audit mode",
    ["tabular", "image"],
    horizontal=True,
    help="Choose tabular for CSV data, image for image datasets (COCO/YOLO)"
)

def _run_tabular_audit(uploaded_file):
    """Run tabular audit on CSV file."""
    bytes_data = uploaded_file.read()
    df = pd.read_csv(StringIO(bytes_data.decode("utf-8")))

    config = AuditConfig(
        target_col=st.session_state.get("target_col"),
        split_col=st.session_state.get("split_col"),
        mode="tabular",
    )
    engine = AuditEngine(df, config)
    result = engine.run()

    report = report_builder.build(
        result,
        source=uploaded_file.name,
        rows=len(df),
        columns=len(df.columns),
        mode="tabular",
    )

    st.session_state["df"] = df
    return report

def _find_dataset_root(extract_dir: Path) -> Path | None:
    """Find dataset root for COCO or YOLO."""

    # COCO
    for p in extract_dir.rglob("annotations.json"):
        return p.parent

    # YOLO (best signal)
    for p in extract_dir.rglob("data.yaml"):
        return p.parent

    # YOLO fallback: detect train/images structure
    for p in extract_dir.rglob("train"):
        if (p / "images").exists():
            return p.parent  # go UP to dataset root

    # YOLO fallback: images/labels pair
    for p in extract_dir.rglob("images"):
        if (p.parent / "labels").exists():
            return p.parent.parent  # go UP twice

    return extract_dir

def _run_image_audit(uploaded_file):
    """Run image audit on uploaded dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        zip_path = tmp_path / "dataset.zip"
        zip_path.write_bytes(uploaded_file.read())

        extract_dir = tmp_path / "images"
        extract_dir.mkdir()

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        dataset_path = _find_dataset_root(extract_dir)
        if dataset_path is None:
            raise ValueError("Could not detect COCO or YOLO dataset")

        st.write("Detected dataset path:", dataset_path)

        loaded = loader.load(dataset_path)

        if isinstance(loaded, tuple):
            dataset, _ = loaded
        else:
            dataset = loaded
        
        config = AuditConfig(
            checks=["integrity", "distribution", "labels", "duplicates", "anomalies"],
            mode="image",
        )
        engine = AuditEngine(dataset, config)
        result = engine.run()

        report = report_builder.build(
            result,
            source=str(dataset_path),
            rows=len(dataset.images),
            columns=len(dataset.categories),
            mode="image",
        )

        st.session_state["image_dataset"] = dataset
        return report

uploaded_file, config, run_button = sidebar.render(mode)

if uploaded_file and run_button:
    try:
        if mode == "tabular":
            result = _run_tabular_audit(uploaded_file)
        else:
            result = _run_image_audit(uploaded_file)

        st.session_state["report"] = result
        st.session_state["mode"] = mode

    except Exception as e:
        st.error(f"Error running audit: {e}")

if "report" in st.session_state:
    report = st.session_state["report"]
    current_mode = st.session_state.get("mode", "tabular")

    score_card.render(report)

    tab1, tab2, tab3 = st.tabs(["Issues", "Distributions", "Modules"])

    with tab1:
        issues_table.render(report)

    with tab2:
        if current_mode == "tabular":
            df = st.session_state.get("df")
            if df is not None:
                distribution_charts.render(df)
            else:
                st.info("Upload a CSV file to see distributions")
        elif current_mode == "image":
            st.info("Image distribution charts available after running image audit")

    with tab3:
        if current_mode == "tabular":
            module_explorer.render(report)
        else:
            st.info("Image Module Explorer")
            image_module_explorer.render(report)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download JSON",
            data=str(report.metadata),
            file_name="report.json",
            mime="application/json",
        )
