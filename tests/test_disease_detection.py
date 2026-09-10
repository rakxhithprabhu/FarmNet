from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
from PIL import Image

from config.settings import DATA_DIR, DISEASE_DATASET_DIR
from modules.disease_detection.preprocessing import (
    dataset_info,
    discover_class_names,
    preprocess_image,
    split_dataset,
)


def test_configured_dataset_root_is_local_color_directory():
    assert DISEASE_DATASET_DIR == DATA_DIR / "color"
    assert DISEASE_DATASET_DIR.name == "color"


def _dataset(tmp_path):
    root = tmp_path / "disease"
    for class_name, color in (("Tomato_healthy", (40, 180, 40)), ("Potato_Late_blight", (180, 40, 40))):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for index in range(8):
            Image.new("RGB", (300, 200), color=color).save(class_dir / f"image_{index}.png")
    return root


def test_dataset_discovery_and_deterministic_class_mapping(tmp_path):
    root = _dataset(tmp_path)
    assert discover_class_names(root) == ["Potato_Late_blight", "Tomato_healthy"]
    resolved_root, class_names, image_count = dataset_info(root)
    assert resolved_root == root.resolve()
    assert class_names == ["Potato_Late_blight", "Tomato_healthy"]
    assert image_count == 16
    first = split_dataset(root, seed=42)
    second = split_dataset(root, seed=42)
    assert first == second
    assert [len(first[index]) for index in (0, 2, 4)] == [10, 2, 4]


def test_image_preprocessing_shape_and_rgb(tmp_path):
    image_path = _dataset(tmp_path) / "Tomato_healthy" / "image_0.png"
    image = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
    processed = preprocess_image(image)
    assert tuple(processed.shape) == (224, 224, 3)


def test_model_output_shape():
    from modules.disease_detection.train import _build_model

    model = _build_model(num_classes=3, weights=None)
    output = model(tf.zeros((1, 224, 224, 3)), training=False)
    assert tuple(output.shape) == (1, 3)
    assert model.get_layer("efficientnetb0_backbone") is not None


def test_saved_model_loading_and_prediction(tmp_path, monkeypatch):
    from modules.disease_detection import predict as prediction_module
    from modules.disease_detection.train import _build_model

    model_path = tmp_path / "disease_model.keras"
    classes_path = tmp_path / "disease_classes.json"
    model = _build_model(num_classes=2, weights=None)
    model.save(model_path)
    classes_path.write_text(json.dumps({"0": "Potato_Late_blight", "1": "Tomato_healthy"}), encoding="utf-8")
    image_path = _dataset(tmp_path) / "Tomato_healthy" / "image_0.png"
    monkeypatch.setattr(prediction_module, "DISEASE_MODEL_PATH", model_path)
    monkeypatch.setattr(prediction_module, "DISEASE_CLASS_NAMES_PATH", classes_path)

    result = prediction_module.predict_disease(str(image_path))
    assert set(("disease", "confidence", "class")) <= result.keys()
    assert result["disease"] in {"Potato_Late_blight", "Tomato_healthy"}
    assert 0.0 <= result["confidence"] <= 1.0

    image_bytes = image_path.read_bytes()
    api_result = prediction_module.predict(image_bytes)
    assert api_result["class"] == api_result["disease"]
