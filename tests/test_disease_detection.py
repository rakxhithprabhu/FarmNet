"""
Tests for the Disease Detection module.

Since we cannot ship the full PlantVillage dataset in CI, these tests
cover the *structural* aspects of the module:
* Transform pipelines produce correct tensor shapes
* Model builder returns the expected architecture
* Predict function raises clearly when no model is saved
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from io import BytesIO

from smart_agriculture.modules.disease_detection.preprocessing import (
    train_transforms,
    val_transforms,
)


class TestTransforms:
    def _dummy_image(self) -> Image.Image:
        return Image.new("RGB", (300, 300), color="green")

    def test_train_transform_shape(self):
        tensor = train_transforms(self._dummy_image())
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_shape(self):
        tensor = val_transforms(self._dummy_image())
        assert tensor.shape == (3, 224, 224)

    def test_transforms_normalize(self):
        tensor = val_transforms(self._dummy_image())
        # After normalization values should NOT all be in [0, 1]
        assert tensor.min() < 0 or tensor.max() > 1


class TestModelBuilder:
    def test_build_model_output_shape(self):
        from smart_agriculture.modules.disease_detection.train import _build_model
        model = _build_model(num_classes=5)
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (1, 5)


class TestPredictGuard:
    def test_predict_raises_without_model(self):
        from smart_agriculture.modules.disease_detection.predict import predict
        img = Image.new("RGB", (224, 224), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        with pytest.raises(FileNotFoundError):
            predict(buf.getvalue())
