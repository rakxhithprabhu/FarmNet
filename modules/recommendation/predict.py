"""Inference for the trained crop recommendation model."""

from __future__ import annotations

import json

import joblib

from config.settings import RECOMMENDATION_METADATA_PATH, RECOMMENDATION_MODEL_PATH
from modules.recommendation.preprocessing import build_feature_frame


def _load_artifacts() -> tuple[dict, dict]:
    bundle = joblib.load(RECOMMENDATION_MODEL_PATH)
    with open(RECOMMENDATION_METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)
    return bundle, metadata


def recommend_crop(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    rainfall: float,
    soil_moisture: float,
    pH: float,
    season: str,
    top_k: int = 3,
) -> dict:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    values = {
        "N": N, "P": P, "K": K, "temperature": temperature,
        "humidity": humidity, "rainfall": rainfall,
        "soil_moisture": soil_moisture, "pH": pH, "season": season,
    }
    bundle, metadata = _load_artifacts()
    frame = build_feature_frame(values, metadata["feature_columns"])
    probabilities = bundle["model"].predict_proba(bundle["preprocessor"].transform(frame))[0]
    classes = metadata["classes"]
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return {
        "recommendations": [
            {"crop": crop, "confidence": round(float(confidence), 6)}
            for crop, confidence in ranked[: min(top_k, len(ranked))]
        ]
    }