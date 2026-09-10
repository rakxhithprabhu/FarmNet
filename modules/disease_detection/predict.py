"""Inference helpers for the EfficientNet-B0 disease classifier."""

from __future__ import annotations

import json
from io import BytesIO

import tensorflow as tf
from PIL import Image

from config.settings import DISEASE_CLASS_NAMES_PATH, DISEASE_MODEL_PATH
from modules.disease_detection.preprocessing import preprocess_image


def _load_model() -> tuple[tf.keras.Model, list[str]]:
    with open(DISEASE_CLASS_NAMES_PATH, encoding="utf-8") as file:
        mapping = json.load(file)
    class_names = [mapping[str(index)] for index in range(len(mapping))] if isinstance(mapping, dict) else mapping
    return tf.keras.models.load_model(DISEASE_MODEL_PATH), class_names


def _predict_image(image: Image.Image) -> dict[str, str | float]:
    model, class_names = _load_model()
    tensor = preprocess_image(tf.convert_to_tensor(image), training=False)
    probabilities = model.predict(tf.expand_dims(tensor, axis=0), verbose=0)[0]
    index = int(tf.argmax(probabilities).numpy())
    result = {"disease": class_names[index], "confidence": round(float(probabilities[index]), 4)}
    return {**result, "class": result["disease"]}


def predict_disease(image_path: str) -> dict[str, str | float]:
    """Classify one leaf image from a filesystem path."""
    with Image.open(image_path) as image:
        return _predict_image(image.convert("RGB"))


def predict(image_bytes: bytes) -> dict[str, str | float]:
    """Classify image bytes for compatibility with the existing API router."""
    with Image.open(BytesIO(image_bytes)) as image:
        return _predict_image(image.convert("RGB"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m modules.disease_detection.predict <image_path>")
    print(json.dumps(predict_disease(sys.argv[1]), indent=2))
