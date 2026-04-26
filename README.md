# DataLint

AI Dataset Quality Auditor — detect, report, and score quality issues in tabular datasets before they break your model.

## Overview

DataLint analyzes raw tabular datasets (CSV/Parquet) and surfaces actionable quality issues with a composite **Data Quality Score (DQS)**. It serves as a gatekeeper between raw data and ML pipelines.

## Features

- **Basic Checks** — Missing values, data type mismatches, constant columns, duplicates
- **Distribution Analysis** — Skewness detection, IQR/Z-score outlier identification
- **Label Auditing** — Class imbalance detection, rare class warnings
- **Leakage Detection** — High correlation with target (Pearson/Spearman), categorical leakage proxies (Cramér's V)
- **Feature Importance** — RF probe model for noise/suspiciously powerful features
- **Duplicate Detection** — Exact duplicates and near-duplicate (cosine similarity/FAISS) matching
- **DQS Scoring** — Weighted composite score (0–100) with severity bands
- **Reports** — JSON, HTML, and CLI output formats

## Installation

```bash
pip install -e .
```

## Quick Start

### CLI

```bash
datalint run dataset.csv --target label --output report.html
```

### Python API

```python
from auditor import AuditEngine, load

df, schema = load("dataset.csv")
config = AuditConfig(target_col="label")
engine = AuditEngine(df, config)
result = engine.run()

print(result.quality_score)  # e.g., 73.4
```

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

## Score Bands

| Score | Band    | Action                     |
|-------|--------|----------------------------|
| 0–40  | Critical | Do not train              |
| 41–65 | Poor    | Major fixes required      |
| 66–80 | Fair    | Minor fixes recommended   |
| 81–100| Good   | Ready for training         |

## Project Structure

```
auditor/
├── core/           # Engine, loader, scorer
├── checks/         # Audit modules (basic, distribution, labels, leakage, importance, dedup)
├── report/         # Report builders and renderers
├── templates/     # HTML report template
└── utils/         # Stats, embeddings, logger

cli/              # Typer CLI entrypoint
app/              # Streamlit UI
app/components/   # UI components
tests/            # Test fixtures and tests
examples/         # Sample datasets and scripts
```

## Tech Stack

- **Core**: Python 3.11+, Pandas, NumPy, SciPy
- **ML**: scikit-learn, FAISS, sentence-transformers
- **CLI**: Typer, Rich
- **UI**: Streamlit, Plotly
- **Reports**: Jinja2

## License

MIT