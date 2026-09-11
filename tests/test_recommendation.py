"""
Tests for the Recommendation Engine.

Covers:
* Rule-based fertiliser selection
* Disease → medicine mapping
* Edge cases (missing inputs)
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from modules.recommendation.engine import (
    Recommendation,
    recommend,
    _classify_yield,
)
from modules.recommendation.model import create_recommendation_model
from modules.recommendation.predict import recommend_crop
from modules.recommendation.preprocessing import (
    build_feature_frame,
    create_preprocessor,
    load_and_validate_dataset,
)


class TestClassifyYield:
    def test_low(self):
        assert _classify_yield(1.0) == "low"

    def test_medium(self):
        assert _classify_yield(3.5) == "medium"

    def test_high(self):
        assert _classify_yield(6.0) == "high"


class TestRecommend:
    def test_known_disease(self):
        rec = recommend(disease_class="Tomato___Late_blight")
        assert isinstance(rec, Recommendation)
        assert "Mancozeb" in rec.medicine or "Metalaxyl" in rec.medicine

    def test_unknown_disease_fallback(self):
        rec = recommend(disease_class="Unknown___Disease")
        assert "agronomist" in rec.medicine.lower()

    def test_no_disease(self):
        rec = recommend(predicted_yield=4.0)
        assert rec.medicine == "No specific medicine recommended"

    def test_low_yield_fertilizer(self):
        rec = recommend(predicted_yield=1.0)
        assert "NPK" in rec.fertilizer

    def test_high_temp_adjustment(self):
        rec = recommend(predicted_yield=3.0, temperature=40.0)
        assert "potassium" in rec.fertilizer.lower()

    def test_high_humidity_adjustment(self):
        rec = recommend(predicted_yield=3.0, humidity=85.0)
        assert "nitrogen" in rec.fertilizer.lower()

    def test_sandy_soil_adjustment(self):
        rec = recommend(predicted_yield=3.0, soil_type="sandy")
        assert "organic" in rec.fertilizer.lower()

    def test_all_inputs(self):
        rec = recommend(
            disease_class="Potato___Late_blight",
            predicted_yield=1.5,
            temperature=36.0,
            humidity=90.0,
            soil_type="sandy",
        )
        assert isinstance(rec, Recommendation)
        assert rec.fertilizer
        assert rec.medicine
        assert rec.notes

    def test_no_inputs(self):
        rec = recommend()
        assert isinstance(rec, Recommendation)
        assert rec.notes == "General recommendation"


def _recommendation_frame() -> pd.DataFrame:
    rows = []
    for index in range(12):
        for crop, season in (("Rice", "Kharif"), ("Wheat", "Rabi"), ("Maize", "Summer")):
            rows.append({
                "SOIL": "Loamy soil" if index % 2 else "Clay soil",
                "SEASON": season,
                "WATER_SOURCE": "irrigated" if index % 2 else "rainfed",
                "SOIL_PH": 6.2 + (index % 3) * 0.1,
                "TEMP": 22 + index % 4,
                "RELATIVE_HUMIDITY": 65 + index,
                "N": 80 + index,
                "P": 35 + index,
                "K": 40 + index,
                "CROPS": crop,
            })
    return pd.DataFrame(rows)


def test_dataset_validation_and_missing_value_preprocessing(tmp_path):
    path = tmp_path / "recommendation.csv"
    frame = _recommendation_frame()
    frame.loc[0, "N"] = None
    frame.to_csv(path, index=False)
    loaded = load_and_validate_dataset(path)
    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(loaded.drop(columns="CROPS"))
    assert transformed.shape[0] == len(frame)
    assert build_feature_frame({
        "soil": "Loamy soil", "season": "Rabi", "water_source": "rainfed",
        "soil_ph": 6.5, "temperature": 20, "humidity": 60,
        "nitrogen": 1, "phosphorus": 2, "potassium": 3,
    }).shape == (1, 9)


def test_missing_required_dataset_column(tmp_path):
    path = tmp_path / "invalid.csv"
    _recommendation_frame().drop(columns="CROPS").to_csv(path, index=False)
    with pytest.raises(ValueError, match="CROPS"):
        load_and_validate_dataset(path)


def test_model_creation_and_training():
    model = create_recommendation_model()
    assert model.n_estimators > 0
    assert model.random_state == 42


def test_training_prediction_top_k_and_saved_loading(tmp_path, monkeypatch):
    from modules.recommendation import predict as prediction_module
    from modules.recommendation import train as training_module

    dataset_path = tmp_path / "recommendation.csv"
    _recommendation_frame().to_csv(dataset_path, index=False)
    model_path = tmp_path / "crop_recommendation.joblib"
    metadata_path = tmp_path / "crop_recommendation_metadata.json"
    monkeypatch.setattr(training_module, "RECOMMENDATION_MODEL_PATH", model_path)
    monkeypatch.setattr(training_module, "RECOMMENDATION_METADATA_PATH", metadata_path)
    monkeypatch.setattr(prediction_module, "RECOMMENDATION_MODEL_PATH", model_path)
    monkeypatch.setattr(prediction_module, "RECOMMENDATION_METADATA_PATH", metadata_path)

    summary = training_module.train(dataset_path)
    assert summary["classes"] == ["Maize", "Rice", "Wheat"]
    assert model_path.exists()
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["target_column"] == "CROPS"

    result = recommend_crop(
        "Loamy soil", "Kharif", "irrigated", 6.5, 25, 75, 90, 40, 40, top_k=3,
    )
    assert len(result["recommendations"]) == 3
    assert {item["crop"] for item in result["recommendations"]} == {"Maize", "Rice", "Wheat"}
    assert all(0 <= item["confidence"] <= 100 for item in result["recommendations"])
    assert all(set(item) == {"crop", "confidence"} for item in result["recommendations"])


def test_invalid_prediction_input():
    with pytest.raises(ValueError, match="top_k"):
        recommend_crop("Loamy soil", "Rabi", "rainfed", 6.5, 20, 60, 1, 2, 3, top_k=0)
    with pytest.raises(ValueError, match="SEASON"):
        build_feature_frame({
            "soil": "Loamy soil", "season": "", "water_source": "rainfed",
            "soil_ph": 6.5, "temperature": 20, "humidity": 60,
            "nitrogen": 1, "phosphorus": 2, "potassium": 3,
        })


def test_crop_recommendation_api(tmp_path, monkeypatch):
    from modules.recommendation import predict as prediction_module
    from modules.recommendation import train as training_module

    dataset_path = tmp_path / "recommendation.csv"
    _recommendation_frame().to_csv(dataset_path, index=False)
    model_path = tmp_path / "crop_recommendation.joblib"
    metadata_path = tmp_path / "crop_recommendation_metadata.json"
    monkeypatch.setattr(training_module, "RECOMMENDATION_MODEL_PATH", model_path)
    monkeypatch.setattr(training_module, "RECOMMENDATION_METADATA_PATH", metadata_path)
    monkeypatch.setattr(prediction_module, "RECOMMENDATION_MODEL_PATH", model_path)
    monkeypatch.setattr(prediction_module, "RECOMMENDATION_METADATA_PATH", metadata_path)
    training_module.train(dataset_path)

    from api.main import app
    response = TestClient(app).post("/api/recommend/crop", json={
        "soil": "Loamy soil", "season": "Kharif", "water_source": "irrigated",
        "soil_ph": 6.5, "N": 90, "P": 40, "K": 40,
        "temperature": 25, "humidity": 75,
    })
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 3
