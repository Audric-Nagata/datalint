# OmniLint

AI Dataset Quality Auditor — detect, report, and score quality issues in tabular and image datasets before they break your model.

## Overview

OmniLint analyzes raw datasets (CSV/Parquet for tabular, COCO/YOLO for images) and surfaces actionable quality issues with a composite **Data Quality Score (DQS)**. It serves as a gatekeeper between raw data and ML pipelines.

## Features

### Tabular Auditing
- **Basic Checks** — Missing values, data type mismatches, constant columns, duplicates
- **Distribution Analysis** — Skewness detection, IQR/Z-score outlier identification
- **Label Auditing** — Class imbalance detection, rare class warnings
- **Leakage Detection** — High correlation with target (Pearson/Spearman), categorical leakage proxies (Cramér's V)
- **Feature Importance** — RF probe model for noise/suspiciously powerful features
- **Duplicate Detection** — Exact duplicates and near-duplicate (cosine similarity/FAISS) matching

### Image Auditing (Beta)
- **Integrity Checks** — Corrupt files, resolution outliers, format inconsistency
- **Distribution Analysis** — Brightness, contrast, color channel imbalance
- **Label Checks** — Label-file mismatch, class imbalance
- **Duplicate Detection** — Perceptual hash (pHash) exact duplicates, CLIP near-duplicates
- **Anomaly Detection** — Blur, exposure issues, blank images

## Installation

### Base (Tabular Only)
```bash
pip install omnilint
```

### With Image Support
```bash
pip install omnilint[image]
```

### Development
```bash
pip install -e ".[dev,image]"
```

## Quick Start

### CLI - Tabular
```bash
OmniLint run dataset.csv --target label --output report.html
```

### CLI - Image
```bash
OmniLint run dataset/ --mode image --format yolo --output report.json
```

### Python API
```python
# Tabular
from omnilint.core import load, AuditEngine
from omnilint.core.engine import AuditConfig

df, schema = load("dataset.csv")
config = AuditConfig(target_col="label")
engine = AuditEngine(df, config)
result = engine.run()

print(result.quality_score)  # e.g., 73.4

# Image
from omnilint.core.loader import load

image_dataset = load("path/to/coco_data")
image_config = AuditConfig(mode="image")
image_engine = AuditEngine(image_dataset, image_config)
image_result = image_engine.run()

print(image_result.quality_score)  # e.g., 85.2
```

### Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

## Input Formats

### Tabular
- CSV (`.csv`)
- Parquet (`.parquet`)

### Image
- COCO JSON (annotations in JSON with images/annotations/categories)
- YOLO (folder with `data.yaml` or `train/images`/`train/labels` structure)

## Score Bands

| Score | Band    | Action                     |
|-------|--------|----------------------------|
| 0–40  | Critical | Do not train              |
| 41–65 | Poor    | Major fixes required      |
| 66–80 | Fair    | Minor fixes recommended   |
| 81–100| Good   | Ready for training         |

## Image Mode Weights

| Module | Weight |
|--------|--------|
| Integrity | 30% |
| Duplicates | 25% |
| Anomalies | 20% |
| Labels | 15% |
| Distribution | 10% |

## Project Structure

```
OmniLint/
├── auditor/
│   ├── core/           # Engine, loader (CSV/COCO/YOLO), scorer
│   ├── tabular/        # Tabular audit modules
│   │   ├── checks/     # basic, distribution, labels, leakage, importance, dedup
│   │   ├── report/     # Report builders
│   │   └── utils/      # Stats, embeddings
│   └── image/          # Image audit modules (beta)
│       ├── checks/     # integrity, distribution, labels, duplicates, anomalies
│       └── utils/      # phash, clip_encoder, pixel_stats
│
├── cli/                # Typer CLI
├── app/                # Streamlit UI
├── app/components/     # Tabular UI components
├── app/components/image/ # Image UI components
└── tests/
    ├── tabular/        # Tabular tests
    └── image/          # Image tests
```

## Tech Stack

### Core
- **Core**: Python 3.11+, Pandas, NumPy, SciPy
- **ML**: scikit-learn, FAISS, sentence-transformers
- **CLI**: Typer, Rich
- **UI**: Streamlit, Plotly
- **Reports**: Jinja2

### Image (Optional)
- **I/O**: Pillow, OpenCV
- **Hashing**: imagehash (pHash)
- **Embeddings**: OpenAI CLIP (ViT-B/32)
- **Blur**: OpenCV Laplacian

## License

MIT