"""
Tests for the Crop Yield Prediction module.

Covers:
* Data loading and preprocessing
* Model training and selection
* Prediction pipeline
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    YIELD_CATEGORICAL_FEATURES,
    YIELD_NUMERIC_FEATURES,
    YIELD_TARGET,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(tmp_path: str, n: int = 200) -> str:
    """Write a tiny synthetic CSV and return its path."""
    rng = np.random.default_rng(0)
    data = {
        "soil_type": rng.choice(["clay", "sandy", "loamy"], n),
        "rainfall_mm": rng.uniform(50, 400, n).round(1),
        "temperature_c": rng.uniform(15, 42, n).round(1),
        "humidity_pct": rng.uniform(30, 95, n).round(1),
        "area_hectares": rng.uniform(0.5, 20, n).round(2),
        "crop_type": rng.choice(["rice", "wheat"], n),
        "season": rng.choice(["kharif", "rabi"], n),
        "yield_tons_per_hectare": rng.uniform(1, 8, n).round(2),
    }
    df = pd.DataFrame(data)
    path = os.path.join(tmp_path, "test_yield.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_load_data_returns_dataframe(self, tmp_path):
        from modules.yield_prediction.preprocessing import load_data
        csv_path = _make_csv(str(tmp_path))
        df = load_data(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert YIELD_TARGET in df.columns

    def test_split_data_shapes(self, tmp_path):
        from modules.yield_prediction.preprocessing import load_data, split_data
        csv_path = _make_csv(str(tmp_path))
        df = load_data(csv_path)
        X_train, X_test, y_train, y_test, preprocessor = split_data(df)
        assert X_train.shape[0] + X_test.shape[0] == len(df)
        assert len(y_train) + len(y_test) == len(df)


class TestTraining:
    def test_train_and_select(self, tmp_path):
        from modules.yield_prediction.train import train_and_select
        csv_path = _make_csv(str(tmp_path))
        # Avoid writing to the real model dir
        summary = train_and_select(csv_path=csv_path, save=False)
        assert "best_model" in summary
        assert "results" in summary
        # At least RF and LR should be present
        assert "RandomForest" in summary["results"]
        assert "LinearRegression" in summary["results"]
        for metrics in summary["results"].values():
            assert "rmse" in metrics
            assert "mae" in metrics
            assert "r2" in metrics


class TestPrediction:
    def test_predict_returns_float(self, tmp_path):
        """Train, save, then predict."""
        import joblib
        from modules.yield_prediction.train import train_and_select
        from modules.yield_prediction.predict import predict
        from config import settings

        csv_path = _make_csv(str(tmp_path))
        # Temporarily redirect model path
        original = settings.YIELD_MODEL_PATH
        settings.YIELD_MODEL_PATH = tmp_path / "test_model.joblib"
        try:
            train_and_select(csv_path=csv_path, save=True)
            result = predict({
                "soil_type": "clay",
                "rainfall_mm": 200.0,
                "temperature_c": 28.0,
                "humidity_pct": 65.0,
                "area_hectares": 2.5,
                "crop_type": "rice",
                "season": "kharif",
            })
            assert isinstance(result, float)
            assert result > 0
        finally:
            settings.YIELD_MODEL_PATH = original
