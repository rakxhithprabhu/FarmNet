"""Two-stage EfficientNet-B0 disease classifier training and evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config.settings import (
    DISEASE_BATCH_SIZE,
    DISEASE_CLASS_NAMES_PATH,
    DISEASE_DATASET_DIR,
    DISEASE_EPOCHS,
    DISEASE_FINE_TUNE_EPOCHS,
    DISEASE_FINE_TUNE_LEARNING_RATE,
    DISEASE_LEARNING_RATE,
    DISEASE_MODEL_PATH,
)
from modules.disease_detection.preprocessing import create_datasets, dataset_info

logger = logging.getLogger(__name__)


def _build_model(num_classes: int, weights: str | None = "imagenet") -> tf.keras.Model:
    """Build the requested EfficientNet-B0 classification architecture."""
    try:
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=(224, 224, 3),
            name="efficientnetb0_backbone",
        )
    except Exception as exc:
        if weights != "imagenet":
            raise
        logger.warning(
            "Could not load EfficientNet-B0 ImageNet weights (%s). "
            "Continuing with randomly initialized weights.",
            exc,
        )
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3),
            name="efficientnetb0_backbone",
        )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = tf.keras.layers.Dropout(0.3, name="head_dropout_1")(features)
    features = tf.keras.layers.Dense(256, activation="relu", name="classifier_dense")(features)
    features = tf.keras.layers.Dropout(0.3, name="head_dropout_2")(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="disease_output")(features)
    return tf.keras.Model(inputs, outputs, name="disease_efficientnet_b0")


def _callbacks() -> list[tf.keras.callbacks.Callback]:
    DISEASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks: list[tf.keras.callbacks.Callback] = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3))
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2))
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            str(DISEASE_MODEL_PATH), monitor="val_accuracy", save_best_only=True
        )
    )
    return callbacks


def train(dataset_root: str | Path = DISEASE_DATASET_DIR, epochs: int = DISEASE_EPOCHS, batch_size: int = DISEASE_BATCH_SIZE, lr: float = DISEASE_LEARNING_RATE, save: bool = True) -> dict:
    """Train in two stages and evaluate the restored best model once on test data."""
    resolved_root, class_names, image_count = dataset_info(dataset_root)
    print(f"Disease dataset:\n{resolved_root}")
    print(f"\nDetected classes: {len(class_names)}")
    print("\n".join(class_names))
    print(f"\nImages detected: {image_count}")
    train_ds, validation_ds, test_ds, class_names = create_datasets(dataset_root, batch_size)
    model = _build_model(len(class_names))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=_callbacks())

    backbone = model.get_layer("efficientnetb0_backbone")
    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=DISEASE_FINE_TUNE_LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=validation_ds, epochs=DISEASE_FINE_TUNE_EPOCHS, callbacks=_callbacks())

    best_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = best_model.predict(test_ds, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision_macro": float(precision_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names))).tolist(),
    }
    print(json.dumps(metrics, indent=2))

    if save:
        with open(DISEASE_CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
            json.dump({str(index): name for index, name in enumerate(class_names)}, file, indent=2)
        logger.info("Model saved to %s", DISEASE_MODEL_PATH)
    return {"num_classes": len(class_names), "class_names": class_names, **metrics}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python -m modules.disease_detection.train [dataset_root]")
    dataset_root = sys.argv[1] if len(sys.argv) == 2 else DISEASE_DATASET_DIR
    print(json.dumps(train(dataset_root), indent=2))
