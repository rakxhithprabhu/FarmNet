"""Train and evaluate the tabular crop recommendation model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config.settings import (
    RECOMMENDATION_DATASET_PATH,
    RECOMMENDATION_METADATA_PATH,
    RECOMMENDATION_MODEL_PATH,
    RECOMMENDATION_RANDOM_STATE,
    RECOMMENDATION_TEST_SIZE,
)
from modules.recommendation.model import create_recommendation_model
from modules.recommendation.preprocessing import (
    TARGET,
    create_preprocessor,
    load_and_validate_dataset,
    split_features_target,
)
from modules.recommendation.utils import feature_importance_table


def train(dataset_path: str | Path = RECOMMENDATION_DATASET_PATH) -> dict:
    """Train, evaluate, and persist the crop recommendation artifacts."""
    frame = load_and_validate_dataset(dataset_path)
    features, target = split_features_target(frame)
    stratify = target if target.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=RECOMMENDATION_TEST_SIZE,
        random_state=RECOMMENDATION_RANDOM_STATE,
        stratify=stratify,
    )

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor(list(features.columns))),
        ("classifier", create_recommendation_model()),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    model = pipeline.named_steps["classifier"]

    classes = [str(value) for value in model.classes_]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=model.classes_, target_names=classes,
            output_dict=True, zero_division=0,
        ),
    }
    importance = feature_importance_table(
        model,
        list(pipeline.named_steps["preprocessor"].get_feature_names_out()),
    )
    print(f"Dataset shape: {frame.shape}")
    print(f"Unique crops: {target.nunique()}")
    print(f"Features: {list(features.columns)}")
    print(f"Missing values: {int(frame.isna().sum().sum())}")
    print(f"Duplicate rows: {int(frame.duplicated().sum())}")
    print(json.dumps(metrics, indent=2))
    print("Feature importance:")
    print(importance.to_string(index=False))

    RECOMMENDATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, RECOMMENDATION_MODEL_PATH)
    metadata = {
        "feature_columns": list(features.columns),
        "target_column": TARGET,
        "classes": classes,
        "transformed_feature_names": list(pipeline.named_steps["preprocessor"].get_feature_names_out()),
        "feature_importance": importance.to_dict(orient="records"),
        "metrics": metrics,
    }
    with open(RECOMMENDATION_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved model path: {RECOMMENDATION_MODEL_PATH}")
    print(f"Saved metadata path: {RECOMMENDATION_METADATA_PATH}")
    return {"dataset": str(Path(dataset_path).resolve()), **metrics, "classes": classes}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        raise SystemExit("Usage: python -m modules.recommendation.train [dataset_path]")
    path = sys.argv[1] if len(sys.argv) == 2 else RECOMMENDATION_DATASET_PATH
    print(json.dumps(train(path), indent=2))