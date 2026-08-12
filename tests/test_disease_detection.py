"""
Tests for the Disease Detection module.

Since we cannot ship the full PlantVillage dataset in CI, these tests
cover the *structural* aspects of the module:
* Transform pipelines produce correct tensor shapes
* Model builder returns the expected architecture
* Predict function raises clearly when no model is saved
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from PIL import Image
from io import BytesIO
from typing import cast

from modules.disease_detection.preprocessing import (
    DiseaseTabularPreprocessor,
    build_label_maps,
    load_disease_data,
    save_preprocessor,
    load_preprocessor,
    train_transforms,
    val_transforms,
)


class TestTransforms:
    def _dummy_image(self) -> Image.Image:
        return Image.new("RGB", (300, 300), color="green")

    def test_train_transform_shape(self):
        tensor = cast(torch.Tensor, train_transforms(self._dummy_image()))
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_shape(self):
        tensor = cast(torch.Tensor, val_transforms(self._dummy_image()))
        assert tensor.shape == (3, 224, 224)

    def test_transforms_normalize(self):
        tensor = cast(torch.Tensor, val_transforms(self._dummy_image()))
        # After normalization values should NOT all be in [0, 1]
        assert tensor.min() < 0 or tensor.max() > 1


class TestModelBuilder:
    def test_build_model_output_shape(self):
        from modules.disease_detection.train import _build_model
        model = _build_model(num_classes=5)
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (1, 5)


class TestMultitaskModel:
    def test_multitask_model_output_shapes(self):
        from modules.disease_detection.train import MultiTaskDiseaseModel

        model = MultiTaskDiseaseModel(
            num_crop_classes=4,
            num_disease_classes=5,
            num_severity_classes=4,
            tabular_input_dim=10,
        )
        dummy_image = torch.randn(2, 3, 224, 224)
        dummy_tabular = torch.randn(2, 10)

        outputs = model(dummy_image, dummy_tabular)
        assert outputs["crop"].shape == (2, 4)
        assert outputs["disease"].shape == (2, 5)
        assert outputs["severity"].shape == (2, 4)


class TestDatasetAndPreprocessing:
    def test_load_disease_data_reads_required_columns(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        pd.DataFrame(
            [
                {
                    "image_name": "img1.jpg",
                    "crop": "Tomato",
                    "disease": "Late blight",
                    "severity": "Moderate",
                    "temperature": 24.0,
                    "humidity": 70.0,
                    "rainfall": 100.0,
                    "season": "Summer",
                    "soil_moisture": 35.0,
                    "pH": 6.5,
                    "N": 50.0,
                    "P": 20.0,
                    "K": 40.0,
                }
            ]
        ).to_csv(csv_path, index=False)

        frame = load_disease_data(str(csv_path))
        assert set(frame.columns) >= {
            "image_name",
            "crop",
            "disease",
            "severity",
            "temperature",
            "humidity",
            "rainfall",
            "season",
            "soil_moisture",
            "pH",
            "N",
            "P",
            "K",
        }

    def test_preprocessor_and_label_maps_are_consistent(self):
        frame = pd.DataFrame(
            [
                {
                    "temperature": 24.0,
                    "humidity": 70.0,
                    "rainfall": 100.0,
                    "season": "Summer",
                    "soil_moisture": 35.0,
                    "pH": 6.5,
                    "N": 50.0,
                    "P": 20.0,
                    "K": 40.0,
                    "crop": "Tomato",
                    "disease": "Late blight",
                    "severity": "Moderate",
                }
            ]
        )
        preprocessor = DiseaseTabularPreprocessor().fit(frame[["temperature", "humidity", "rainfall", "season", "soil_moisture", "pH", "N", "P", "K"]])
        transformed = preprocessor.transform(frame[["temperature", "humidity", "rainfall", "season", "soil_moisture", "pH", "N", "P", "K"]])
        assert transformed.shape[1] == preprocessor.output_dim
        assert build_label_maps(frame)["disease"]["Late blight"] == 0

    def test_preprocessor_round_trip(self, tmp_path):
        frame = pd.DataFrame(
            [
                {
                    "temperature": 24.0,
                    "humidity": 70.0,
                    "rainfall": 100.0,
                    "season": "Summer",
                    "soil_moisture": 35.0,
                    "pH": 6.5,
                    "N": 50.0,
                    "P": 20.0,
                    "K": 40.0,
                }
            ]
        )
        preprocessor = DiseaseTabularPreprocessor().fit(frame)
        path = tmp_path / "preprocessor.json"
        save_preprocessor(preprocessor, path)
        loaded = load_preprocessor(path)
        assert loaded.output_dim == preprocessor.output_dim


class TestPredictGuard:
    def test_predict_raises_without_model(self):
        from modules.disease_detection.predict import predict
        img = Image.new("RGB", (224, 224), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        with pytest.raises(FileNotFoundError):
            predict(buf.getvalue())
