"""
Inference helper for the Crop Disease Detection module.

Loads the saved ResNet-18 model and returns the predicted disease class
for a given leaf image.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import models
import torch.nn as nn

from smart_agriculture.config.settings import (
    DISEASE_CLASS_NAMES_PATH,
    DISEASE_MODEL_PATH,
)
from smart_agriculture.modules.disease_detection.preprocessing import val_transforms


def _load_model() -> tuple[nn.Module, list[str]]:
    """Load saved model weights and class names."""
    with open(DISEASE_CLASS_NAMES_PATH) as f:
        class_names: list[str] = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(DISEASE_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, class_names


def predict(image_bytes: bytes) -> dict[str, str | float]:
    """
    Classify a crop leaf image.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a JPEG / PNG image.

    Returns
    -------
    dict
        ``{"class": "Tomato___Late_blight", "confidence": 0.97}``
    """
    model, class_names = _load_model()

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = val_transforms(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = probs.max(dim=1)

    return {
        "class": class_names[idx.item()],
        "confidence": round(confidence.item(), 4),
    }
