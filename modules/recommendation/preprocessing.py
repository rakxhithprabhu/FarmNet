"""Validation and leakage-safe preprocessing for crop recommendation data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from config.settings import RECOMMENDATION_DATASET_PATH

NUMERICAL_FEATURES = ["SOIL_PH", "TEMP", "RELATIVE_HUMIDITY", "N", "P", "K"]
CATEGORICAL_FEATURES = ["SOIL", "SEASON", "WATER_SOURCE"]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "CROPS"

# Compatibility aliases for callers that imported the previous names.
FEATURE_COLUMNS = FEATURES
TARGET_COLUMN = TARGET


def load_and_validate_dataset(
    dataset_path: str | Path = RECOMMENDATION_DATASET_PATH,
) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Crop recommendation dataset not found: {path}. "
            "Provide a real tabular dataset at the configured path."
        )
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Crop recommendation dataset is missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise ValueError(f"Crop recommendation dataset is empty: {path}")
    if frame[TARGET].isna().any():
        raise ValueError(f"Crop recommendation target column '{TARGET}' cannot contain missing values")
    return frame[FEATURES + [TARGET]].copy()


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame[FEATURES].copy(), frame[TARGET].astype(str)


def create_preprocessor(feature_columns: list[str] | None = None) -> ColumnTransformer:
    feature_columns = feature_columns or FEATURES
    numeric_features = [column for column in NUMERICAL_FEATURES if column in feature_columns]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
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
    feature_columns = feature_columns or FEATURES
    aliases = {
        "soil": "SOIL", "season": "SEASON", "water_source": "WATER_SOURCE",
        "soil_ph": "SOIL_PH", "temperature": "TEMP", "humidity": "RELATIVE_HUMIDITY",
        "nitrogen": "N", "phosphorus": "P", "potassium": "K",
    }
    values = {**values, **{target: values[source] for source, target in aliases.items() if source in values}}
    missing = [column for column in feature_columns if column not in values or values[column] is None]
    if missing:
        raise ValueError(f"Missing recommendation inputs: {', '.join(missing)}")
    frame = pd.DataFrame([{column: values[column] for column in feature_columns}])
    for column in feature_columns:
        if column in CATEGORICAL_FEATURES:
            continue
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Recommendation input '{column}' must be numeric") from exc
    for column in CATEGORICAL_FEATURES:
        if column in frame:
            frame[column] = frame[column].astype(str).str.strip()
            if not frame[column].iloc[0]:
                raise ValueError(f"Recommendation input '{column}' cannot be empty")
    return frame