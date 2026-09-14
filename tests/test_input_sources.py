from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_recommendation_merges_retrieved_weather(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "api.routers.recommendation_router.get_weather",
        lambda location: {"temperature": 24.0, "humidity": 70.0, "rainfall": 12.0},
    )

    def fake_recommend_crop(**values):
        captured.update(values)
        return {"recommendations": [{"crop": "rice", "confidence": 0.9}]}

    monkeypatch.setattr("api.routers.recommendation_router.recommend_crop", fake_recommend_crop)
    response = TestClient(app).post("/api/recommend/crop", json={
        "N": 90, "P": 40, "K": 40, "pH": 6.5, "season": "Kharif",
        "soil_moisture": 65, "location": "Nairobi",
    })
    assert response.status_code == 200
    assert captured["temperature"] == 24.0
    assert captured["rainfall"] == 12.0
    assert response.json()["weather"]["humidity"] == 70.0


def test_weather_failure_is_user_safe(monkeypatch):
    from api.services.weather import WeatherServiceError

    def fail(_location):
        raise WeatherServiceError("Unable to retrieve weather.")

    monkeypatch.setattr("api.routers.yield_router.get_weather", fail)
    response = TestClient(app).post("/api/yield/predict", json={
        "soil_type": "clay", "area_hectares": 2, "crop_type": "rice",
        "season": "kharif", "location": "Nairobi",
    })
    assert response.status_code == 503
    assert response.json()["detail"] == "Unable to retrieve weather."


def test_disease_rejects_non_image_upload():
    response = TestClient(app).post(
        "/api/disease/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
