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
from scipy import sparse
from typing import Any

from config.settings import (
    YIELD_CATEGORICAL_FEATURES,
    YIELD_NUMERIC_FEATURES,
    YIELD_TARGET,
    YIELD_DATASET_PATH,
    YIELD_TEST_SIZE,
    YIELD_RANDOM_STATE,
    YIELD_DATE_FEATURE,
    YIELD_IGNORE_COLUMNS,
)


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the crop-yield CSV and return a cleaned DataFrame."""
    path = path or str(YIELD_DATASET_PATH)
    df = pd.read_csv(path)
    # -----------------------------
# Date Feature Engineering
# -----------------------------
    if YIELD_DATE_FEATURE in df.columns:
        df[YIELD_DATE_FEATURE] = pd.to_datetime(
            df[YIELD_DATE_FEATURE],
            errors="coerce"
        )

        df["month"] = df[YIELD_DATE_FEATURE].dt.month
        df["day_of_year"] = df[YIELD_DATE_FEATURE].dt.dayofyear

        # Optional season feature
        df["season"] = df["month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Summer", 4: "Summer", 5: "Summer",
            6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
            9: "Autumn", 10: "Autumn", 11: "Autumn"
        })

    # Drop unused columns
    df.drop(columns=YIELD_IGNORE_COLUMNS, inplace=True, errors="ignore")

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
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, YIELD_NUMERIC_FEATURES),
            ("cat", categorical_transformer, YIELD_CATEGORICAL_FEATURES),
        ]
    )


def _to_dense(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Convert sparse transformer output to a dense NumPy array."""
    if sparse.issparse(matrix):
        sparse_matrix: Any = matrix
        return np.asarray(sparse_matrix.toarray())
    return np.asarray(matrix)


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
    X_train = _to_dense(preprocessor.fit_transform(X_train))
    X_test = _to_dense(preprocessor.transform(X_test))

    return X_train, X_test, y_train, y_test, preprocessor
