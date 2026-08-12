"""Inference helpers for the multimodal disease detection model."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from PIL import Image

from config.settings import (
    DISEASE_CLASS_NAMES_PATH,
    DISEASE_METADATA_PATH,
    DISEASE_MODEL_PATH,
)
from modules.disease_detection.preprocessing import (
    DiseaseTabularPreprocessor,
    load_preprocessor,
    val_transforms,
)
from modules.disease_detection.train import MultiTaskDiseaseModel


def _load_model() -> tuple[torch.nn.Module, dict[str, Any], DiseaseTabularPreprocessor]:
    """Load the trained model, metadata, and fitted tabular preprocessor."""
    if not DISEASE_MODEL_PATH.exists():
        raise FileNotFoundError("Disease model not trained yet.")
    if not DISEASE_METADATA_PATH.exists():
        raise FileNotFoundError("Disease model metadata not found.")

    with DISEASE_METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    preprocessor = load_preprocessor(DISEASE_METADATA_PATH.parent / "preprocessor.json")
    model = MultiTaskDiseaseModel(
        num_crop_classes=len(metadata["crop_names"]),
        num_disease_classes=len(metadata["disease_names"]),
        num_severity_classes=len(metadata["severity_names"]),
        tabular_input_dim=preprocessor.output_dim,
    )
    state = torch.load(DISEASE_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, metadata, preprocessor


def _prepare_image(image: Any) -> torch.Tensor:
    if isinstance(image, (bytes, bytearray)):
        img = Image.open(BytesIO(bytes(image))).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        raise TypeError("Expected image bytes, a PIL image, or a filesystem path")
    image_tensor = cast(torch.Tensor, val_transforms(img))
    return image_tensor.unsqueeze(0)


def _prepare_tabular(features: dict[str, Any], preprocessor: DiseaseTabularPreprocessor, metadata: dict[str, Any] | None = None) -> np.ndarray:
    ordered_columns = metadata.get("feature_columns", list(features.keys())) if metadata is not None else list(features.keys())
    ordered_row = {column: features.get(column, np.nan) for column in ordered_columns}
    frame = pd.DataFrame([ordered_row], columns=ordered_columns)
    return preprocessor.transform(frame)


def _predict_single(
    image: Any,
    features: dict[str, Any],
    model: torch.nn.Module,
    metadata: dict[str, Any],
    preprocessor: DiseaseTabularPreprocessor,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_tensor = _prepare_image(image).to(device)
    tabular_array = _prepare_tabular(features, preprocessor, metadata)
    tabular_tensor = torch.from_numpy(tabular_array).to(device).float()

    with torch.no_grad():
        outputs = model(image_tensor, tabular_tensor)

    def _decode(logits: torch.Tensor, names: list[str]) -> tuple[str, float]:
        probs = torch.softmax(logits, dim=1)
        confidence, index = probs.max(dim=1)
        index_value = int(index.item())
        label = names[index_value]
        return label, round(float(confidence.item()), 4)

    crop, crop_confidence = _decode(outputs["crop"], metadata["crop_names"])
    disease, disease_confidence = _decode(outputs["disease"], metadata["disease_names"])
    severity, severity_confidence = _decode(outputs["severity"], metadata["severity_names"])

    return {
        "crop": crop,
        "crop_confidence": crop_confidence,
        "disease": disease,
        "disease_confidence": disease_confidence,
        "severity": severity,
        "severity_confidence": severity_confidence,
    }


def predict(image_bytes: bytes) -> dict[str, Any]:
    """Backward-compatible single-image prediction helper."""
    model, metadata, preprocessor = _load_model()
    features = {
        "temperature": 25.0,
        "humidity": 60.0,
        "rainfall": 100.0,
        "season": "Summer",
        "soil_moisture": 30.0,
        "pH": 6.5,
        "N": 50.0,
        "P": 20.0,
        "K": 40.0,
    }
    return _predict_single(image_bytes, features, model, metadata, preprocessor)


def predict_disease(
    image: Any,
    temperature: float,
    humidity: float,
    rainfall: float,
    season: str,
    soil_moisture: float,
    pH: float,
    N: float,
    P: float,
    K: float,
) -> dict[str, Any]:
    """Predict all three disease outputs from one image and the tabular features."""
    model, metadata, preprocessor = _load_model()
    features = {
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "season": season,
        "soil_moisture": soil_moisture,
        "pH": pH,
        "N": N,
        "P": P,
        "K": K,
    }
    return _predict_single(image, features, model, metadata, preprocessor)


def predict_batch(
    images: list[Any],
    features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Predict for multiple images with corresponding tabular feature rows."""
    model, metadata, preprocessor = _load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if len(images) != len(features):
        raise ValueError("The number of images and feature rows must match.")

    results: list[dict[str, Any]] = []
    for image, row in zip(images, features):
        results.append(_predict_single(image, row, model, metadata, preprocessor))
    return results
