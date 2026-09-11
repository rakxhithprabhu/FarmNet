"""
Inference helper for the Crop Yield Prediction module.

Loads the persisted model + preprocessor and exposes a ``predict`` function
that accepts a dict of raw feature values.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    YIELD_CATEGORICAL_FEATURES,
    YIELD_MODEL_PATH,
    YIELD_NUMERIC_FEATURES,
)


def _load_artifacts() -> tuple[Any, Any]:
    """Load the saved model and preprocessor."""
    artifacts = joblib.load(YIELD_MODEL_PATH)
    return artifacts["model"], artifacts["preprocessor"]


def predict(features: dict[str, Any]) -> float:
    """
    Predict crop yield given raw input features.

    Parameters
    ----------
    features : dict
        Must contain keys matching YIELD_NUMERIC_FEATURES and
        YIELD_CATEGORICAL_FEATURES.

    Returns
    -------
    float
        Predicted yield in tons / hectare.
    """
    model, preprocessor = _load_artifacts()

    row = pd.DataFrame(
        [features],
        columns=YIELD_NUMERIC_FEATURES + YIELD_CATEGORICAL_FEATURES,
    )
    X = preprocessor.transform(row)
    prediction = model.predict(X)
    return float(prediction[0])
