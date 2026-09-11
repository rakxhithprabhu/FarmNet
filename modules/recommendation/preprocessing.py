"""Validation and leakage-safe preprocessing for crop recommendation data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "rainfall",
    "soil_moisture",
    "pH",
    "season",
]
TARGET_COLUMN = "crop"
COMMON_FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if not columns:
        raise ValueError("No supported crop recommendation feature columns were found")
    return columns


def load_and_validate_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Crop recommendation dataset not found: {path}. "
            "Provide a real tabular dataset at the configured path."
        )
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"ph": "pH", "label": "crop"})
    feature_columns = _feature_columns(frame)
    missing = [column for column in COMMON_FEATURE_COLUMNS + [TARGET_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Crop recommendation dataset is missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise ValueError(f"Crop recommendation dataset is empty: {path}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Crop recommendation target column 'crop' cannot contain missing values")
    return frame[feature_columns + [TARGET_COLUMN]].copy()


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN].astype(str)


def create_preprocessor(feature_columns: list[str] | None = None) -> ColumnTransformer:
    feature_columns = feature_columns or FEATURE_COLUMNS
    numeric_features = [column for column in feature_columns if column != "season"]
    categorical_features = [column for column in feature_columns if column == "season"]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_frame(values: dict, feature_columns: list[str] | None = None) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    values = {**values, "pH": values.get("pH", values.get("ph"))}
    missing = [column for column in feature_columns if column not in values or values[column] is None]
    if missing:
        raise ValueError(f"Missing recommendation inputs: {', '.join(missing)}")
    frame = pd.DataFrame([{column: values[column] for column in feature_columns}])
    for column in feature_columns:
        if column == "season":
            continue
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Recommendation input '{column}' must be numeric") from exc
    if "season" in frame:
        frame["season"] = frame["season"].astype(str).str.strip()
    if "season" in frame and not frame["season"].iloc[0]:
        raise ValueError("Recommendation input 'season' cannot be empty")
    return frame