"""Data loading and schema inference."""

from __future__ import annotations

import json
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Literal


@dataclass
class Schema:
    source: str
    rows: int
    columns: int
    schema: dict


@dataclass
class ImageEntry:
    filename: str
    width: int
    height: int
    file_path: Path
    annotations: list[dict] = field(default_factory=list)
    split: str = "train"


@dataclass
class ImageDataset:
    format: Literal["coco", "yolo"]
    images: list[ImageEntry]
    categories: list[str]
    split_col: pd.Series | None = None
    root_path: Path | None = None


def infer_schema(df: pd.DataFrame) -> dict:
    """Infer column types and nullable status."""
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        nullable = df[col].isna().any()
        if pd.api.types.is_numeric_dtype(dtype):
            inferred = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            inferred = "datetime"
        else:
            inferred = "categorical"
        schema[col] = {
            "inferred_type": inferred,
            "nullable": nullable,
        }
    return schema


def detect_format(source: Path) -> Literal["csv", "parquet", "coco", "yolo"]:
    """Detect input format from path."""
    if source.is_dir():
        if (source / "annotations.json").exists() or any(f.suffix == ".json" for f in source.glob("*.json")):
            return "coco"
        if (source / "data.yaml").exists() or (source / "train").exists():
            return "yolo"
        if any((source / "train").rglob("*.txt").__iter__()):
            return "yolo"
    elif source.suffix.lower() == ".csv":
        return "csv"
    elif source.suffix.lower() == ".parquet":
        return "parquet"
    raise ValueError(f"Cannot detect format for: {source}")


def load_coco(root: Path) -> ImageDataset:
    """Load COCO format dataset."""
    json_files = list(root.glob("*.json")) + list(root.glob("*/instances_*.json"))
    if not json_files:
        raise ValueError(f"No COCO annotation file found in {root}")
    
    ann_file = json_files[0]
    with open(ann_file) as f:
        coco = json.load(f)
    
    images = []
    img_map = {img["id"]: img for img in coco.get("images", [])}
    
    for ann in coco.get("annotations", []):
        img_info = img_map.get(ann["image_id"])
        if not img_info:
            continue
        
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]
        img_path = root / file_name if Path(file_name).is_absolute() else root / file_name
        
        images.append(ImageEntry(
            filename=file_name,
            width=width,
            height=height,
            file_path=img_path,
            annotations=[{
                "category_id": ann["category_id"],
                "bbox": ann.get("bbox", []),
                "area": ann.get("area"),
            }],
            split="train",
        ))
    
    categories = [cat["name"] for cat in coco.get("categories", [])]
    
    return ImageDataset(
        format="coco",
        images=images,
        categories=categories,
        split_col=None,
        root_path=root,
    )


def load_yolo(root: Path) -> ImageDataset:
    """Load YOLO format dataset."""
    if (root / "data.yaml").exists():
        try:
            import yaml
            with open(root / "data.yaml") as f:
                config = yaml.safe_load(f)
                train_path = config.get("train", "train/images")
                val_path = config.get("val", "val/images")
        except ImportError:
            train_path = "train/images"
            val_path = "val/images"
    else:
        train_path = "train/images"
        val_path = "val/images"
    
    images = []
    categories = set()
    splits = {"train": root / train_path, "val": root / val_path}
    
    for split_name, split_path in splits.items():
        img_dir = split_path if split_path.is_absolute() else root / split_path
        label_dir = img_dir.parent / "labels"
        
        if not img_dir.exists():
            continue
        
        for img_file in img_dir.glob("*.*"):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                width, height = 640, 640
                ann_file = label_dir / f"{img_file.stem}.txt"
                annotations = []
                
                if ann_file.exists():
                    with open(ann_file) as af:
                        for line in af:
                            parts = line.strip().split()
                            if parts:
                                cat_id = int(parts[0])
                                cx, cy, w, h = map(float, parts[1:5])
                                annotations.append({
                                    "category_id": cat_id,
                                    "cx": cx, "cy": cy, "w": w, "h": h,
                                })
                                categories.add(cat_id)
                
                images.append(ImageEntry(
                    filename=img_file.name,
                    width=width,
                    height=height,
                    file_path=img_file,
                    annotations=annotations,
                    split=split_name,
                ))
    
    cat_list = sorted(list(categories))
    cat_names = [f"class_{i}" for i in cat_list]
    
    return ImageDataset(
        format="yolo",
        images=images,
        categories=cat_names,
        split_col=pd.Series([img.split for img in images]),
        root_path=root,
    )


def load(source: Union[str, Path, pd.DataFrame]) -> tuple[pd.DataFrame, Schema] | ImageDataset:
    """Load data from file path or DataFrame. Auto-detects tabular or image format."""
    if isinstance(source, pd.DataFrame):
        return _load_tabular_df(source), Schema(
            source="dataframe",
            rows=len(source),
            columns=len(source.columns),
            schema=infer_schema(source),
        )
    
    path = Path(source)
    
    if path.is_file():
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            return _load_tabular_df(df), Schema(
                source=path.name,
                rows=len(df),
                columns=len(df.columns),
                schema=infer_schema(df),
            )
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            return _load_tabular_df(df), Schema(
                source=path.name,
                rows=len(df),
                columns=len(df.columns),
                schema=infer_schema(df),
            )
        elif path.suffix.lower() == ".json":
            return load_coco(path.parent)
    
    fmt = detect_format(path)
    
    if fmt == "coco":
        return load_coco(path)
    elif fmt == "yolo":
        return load_yolo(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _load_tabular_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return tabular DataFrame."""
    if df.empty or df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns and 1 row")
    return df


def validate_minimum_requirements(df: pd.DataFrame) -> None:
    """Raise if dataset doesn't meet minimum requirements."""
    if df.empty:
        raise ValueError("Dataset is empty")
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns")