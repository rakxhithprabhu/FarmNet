"""Inference for the trained crop recommendation model."""

from __future__ import annotations

import json

import joblib

from config.settings import RECOMMENDATION_METADATA_PATH, RECOMMENDATION_MODEL_PATH
from modules.recommendation.preprocessing import FEATURES, build_feature_frame


def _load_artifacts() -> tuple[object, dict]:
    pipeline = joblib.load(RECOMMENDATION_MODEL_PATH)
    with open(RECOMMENDATION_METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)
    return pipeline, metadata


def predict_top_crops(pipeline, frame, top_k: int = 3) -> list[dict[str, str | float]]:
    probabilities = pipeline.predict_proba(frame)[0]
    classes = pipeline.classes_
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return [
        {"crop": str(crop), "confidence": round(float(confidence) * 100, 2)}
        for crop, confidence in ranked[: min(top_k, len(ranked))]
    ]


def recommend_crop(
    soil: str,
    season: str,
    water_source: str,
    soil_ph: float,
    temperature: float,
    humidity: float,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    top_k: int = 3,
) -> dict:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    values = {
        "soil": soil, "season": season, "water_source": water_source,
        "soil_ph": soil_ph, "temperature": temperature, "humidity": humidity,
        "nitrogen": nitrogen, "phosphorus": phosphorus, "potassium": potassium,
    }
    pipeline, _ = _load_artifacts()
    frame = build_feature_frame(values, FEATURES)
    recommendations = predict_top_crops(pipeline, frame, top_k)
    return {
        "crop": recommendations[0]["crop"],
        "recommendations": recommendations,
    }