"""
Data preprocessing pipeline for the Crop Yield Prediction module.

Responsibilities
----------------
* Load CSV data
* Handle missing values
* Encode categorical features (OneHot)
* Scale numeric features (StandardScaler)
* Train / test split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import (
    YIELD_CATEGORICAL_FEATURES,
    YIELD_DATASET_PATH,
    YIELD_NUMERIC_FEATURES,
    YIELD_TARGET,
    YIELD_TEST_SIZE,
    YIELD_RANDOM_STATE,
)


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the crop-yield CSV and return a cleaned DataFrame."""
    path = path or str(YIELD_DATASET_PATH)
    df = pd.read_csv(path)

    # Basic cleaning
    df.dropna(subset=[YIELD_TARGET], inplace=True)
    for col in YIELD_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in YIELD_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer for numeric + categorical cols."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, YIELD_NUMERIC_FEATURES),
            ("cat", categorical_transformer, YIELD_CATEGORICAL_FEATURES),
        ]
    )


def split_data(
    df: pd.DataFrame,
    test_size: float = YIELD_TEST_SIZE,
    random_state: int = YIELD_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Split data into train/test and return preprocessed arrays.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor (fitted)
    """
    X = df[YIELD_NUMERIC_FEATURES + YIELD_CATEGORICAL_FEATURES]
    y = df[YIELD_TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor
