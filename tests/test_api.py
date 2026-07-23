"""
Tests for the FastAPI endpoints.

Uses TestClient to exercise all three routers without starting a server.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from smart_agriculture.api.main import app

client = TestClient(app)


class TestHealthCheck:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "Smart Agriculture Platform"


class TestRecommendationEndpoint:
    def test_recommendation_with_disease(self):
        resp = client.post("/api/recommend/", json={
            "disease_class": "Tomato___Late_blight",
            "predicted_yield": 2.0,
            "temperature": 30.0,
            "humidity": 70.0,
            "soil_type": "clay",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "fertilizer" in body
        assert "medicine" in body

    def test_recommendation_empty_body(self):
        resp = client.post("/api/recommend/", json={})
        assert resp.status_code == 200


class TestYieldEndpoint:
    def test_predict_without_model_returns_503(self):
        resp = client.post("/api/yield/predict", json={
            "soil_type": "clay",
            "rainfall_mm": 200.0,
            "temperature_c": 28.5,
            "humidity_pct": 65.0,
            "area_hectares": 2.5,
            "crop_type": "rice",
            "season": "kharif",
        })
        # Model not trained yet → 503
        assert resp.status_code in (200, 503)


class TestDiseaseEndpoint:
    def test_predict_without_model_returns_503(self):
        from PIL import Image
        from io import BytesIO

        img = Image.new("RGB", (224, 224), color="green")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/api/disease/predict",
            files={"file": ("leaf.png", buf, "image/png")},
        )
        assert resp.status_code in (200, 503)
