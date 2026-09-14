"""Train and evaluate the tabular crop recommendation model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

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
from modules.recommendation.model import create_recommendation_model, create_recommendation_search
from modules.recommendation.preprocessing import (
    TARGET,
    create_preprocessor,
    load_and_validate_dataset,
    split_features_target,
)
from modules.recommendation.utils import (
    cross_validation_summary,
    feature_importance_table,
    robustness_stability,
)


def _evaluate(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    predictions = pipeline.predict(features)
    classes = [str(value) for value in pipeline.named_steps["classifier"].classes_]
    classifier_classes = pipeline.named_steps["classifier"].classes_
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision_macro": float(precision_score(target, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(target, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(target, predictions, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(target, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(target, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(target, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(target, predictions, labels=classifier_classes).tolist(),
        "classification_report": classification_report(
            target, predictions, labels=classifier_classes, target_names=classes,
            output_dict=True, zero_division=0,
        ),
    }


def train(
    dataset_path: str | Path = RECOMMENDATION_DATASET_PATH,
    *,
    tune: bool = False,
    tuning_iter: int = 20,
    tuning_cv: int = 3,
    enhanced_validation: bool = False,
    generate_reports: bool = False,
    report_dir: str | Path = "reports/recommendation",
) -> dict:
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

    baseline_pipeline = Pipeline([
        ("preprocessor", create_preprocessor(list(features.columns))),
        ("classifier", create_recommendation_model()),
    ])
    baseline_pipeline.fit(x_train, y_train)
    baseline_metrics = _evaluate(baseline_pipeline, x_test, y_test)
    pipeline = baseline_pipeline
    best_params = {}
    best_cv_score = None
    tuned_metrics = None
    if tune:
        search = create_recommendation_search(baseline_pipeline, n_iter=tuning_iter, cv=tuning_cv)
        search.fit(x_train, y_train)
        tuned_pipeline = cast(Pipeline, search.best_estimator_)
        tuned_metrics = _evaluate(tuned_pipeline, x_test, y_test)
        best_params = search.best_params_
        best_cv_score = float(search.best_score_)
        if tuned_metrics["f1_macro"] > baseline_metrics["f1_macro"]:
            pipeline = tuned_pipeline

    metrics = _evaluate(pipeline, x_test, y_test)
    model = pipeline.named_steps["classifier"]

    classes = [str(value) for value in model.classes_]
    importance = feature_importance_table(
        model,
        list(pipeline.named_steps["preprocessor"].get_feature_names_out()),
    )
    print(f"Dataset shape: {frame.shape}")
    print(f"Unique crops: {target.nunique()}")
    print(f"Features: {list(features.columns)}")
    print(f"Missing values: {int(frame.isna().sum().sum())}")
    print(f"Duplicate rows: {int(frame.duplicated().sum())}")
    feature_label_conflicts = int(
        frame.groupby(list(features.columns))[TARGET].nunique().gt(1).sum()
    )
    print(f"Duplicate feature vectors with conflicting targets: {feature_label_conflicts}")
    print(f"Baseline metrics: accuracy={baseline_metrics['accuracy']:.6f}, f1_macro={baseline_metrics['f1_macro']:.6f}")
    if tuned_metrics is not None:
        print(f"Tuned metrics: accuracy={tuned_metrics['accuracy']:.6f}, f1_macro={tuned_metrics['f1_macro']:.6f}")
        print(f"Best CV macro F1: {best_cv_score:.6f}")
        print(f"Best hyperparameters: {best_params}")
    print(json.dumps(metrics, indent=2))
    print("Feature importance:")
    print(importance.to_string(index=False))

    validation = {}
    if enhanced_validation:
        validation["cross_validation"] = cross_validation_summary(x_train, y_train, n_splits=5)
        validation["robustness"] = robustness_stability(pipeline, x_test)
        print(f"Cross-validation: {validation['cross_validation']}")
        print(f"Robustness: {validation['robustness']}")

    shap_summary = None
    if generate_reports:
        from modules.recommendation.explainability import generate_shap_reports

        shap_summary = generate_shap_reports(pipeline, x_test, report_dir)
        print(f"SHAP reports saved to: {report_dir}")

    RECOMMENDATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, RECOMMENDATION_MODEL_PATH)
    metadata = {
        "feature_columns": list(features.columns),
        "target_column": TARGET,
        "classes": classes,
        "transformed_feature_names": list(pipeline.named_steps["preprocessor"].get_feature_names_out()),
        "feature_importance": importance.to_dict(orient="records"),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "data_quality": {
            "missing_values": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "conflicting_feature_vectors": feature_label_conflicts,
            "constant_columns": [column for column in frame.columns if frame[column].nunique(dropna=False) <= 1],
            "class_distribution": target.value_counts().to_dict(),
        },
        "validation": validation,
    }
    with open(RECOMMENDATION_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved model path: {RECOMMENDATION_MODEL_PATH}")
    print(f"Saved metadata path: {RECOMMENDATION_METADATA_PATH}")
    return {
        "dataset": str(Path(dataset_path).resolve()), **metrics, "classes": classes,
        "baseline_metrics": baseline_metrics, "tuned_metrics": tuned_metrics,
        "best_params": best_params, "best_cv_score": best_cv_score,
        "validation": validation, "shap_reports": bool(shap_summary),
    }


if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", nargs="?", default=RECOMMENDATION_DATASET_PATH)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--enhanced-validation", action="store_true")
    parser.add_argument("--reports", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(train(
        arguments.dataset_path,
        tune=arguments.tune,
        enhanced_validation=arguments.enhanced_validation,
        generate_reports=arguments.reports,
    ), indent=2, default=str))