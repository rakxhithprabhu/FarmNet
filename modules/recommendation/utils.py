"""Explainability helpers for the crop recommendation model."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from modules.recommendation.model import create_recommendation_model
from modules.recommendation.preprocessing import FEATURES, create_preprocessor


def feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        raise ValueError("The supplied model does not expose feature_importances_")
    if len(feature_names) != len(model.feature_importances_):
        raise ValueError("Feature-name count does not match model importances")
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def create_recommendation_pipeline() :
    """Build a fresh preprocessing plus baseline classifier Pipeline."""
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("preprocessor", create_preprocessor(FEATURES)),
        ("classifier", create_recommendation_model()),
    ])


def cross_validation_summary(features: pd.DataFrame, target: pd.Series, n_splits: int = 5) -> dict:
    """Evaluate accuracy and macro F1 with stratified folds on training data."""
    scores = cross_validate(
        create_recommendation_pipeline(),
        features,
        target,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=1,
        return_train_score=False,
    )
    return {
        metric: {
            "mean": float(scores[f"test_{metric}"].mean()),
            "std": float(scores[f"test_{metric}"].std()),
            "min": float(scores[f"test_{metric}"].min()),
            "max": float(scores[f"test_{metric}"].max()),
        }
        for metric in ("accuracy", "f1_macro")
    }


def robustness_stability(pipeline, features: pd.DataFrame, sample_size: int = 100) -> dict:
    """Measure prediction stability under +/-5% and +/-10% numeric perturbations."""
    from modules.recommendation.preprocessing import NUMERICAL_FEATURES

    sample = features.sample(min(sample_size, len(features)), random_state=42).copy()
    original = pipeline.predict(sample)
    stable = 0
    total = 0
    for feature in NUMERICAL_FEATURES:
        for rate in (-0.10, -0.05, 0.05, 0.10):
            perturbed = sample.copy()
            perturbed[feature] = perturbed[feature] * (1 + rate)
            predictions = pipeline.predict(perturbed)
            stable += int((predictions == original).sum())
            total += len(predictions)
    return {
        "stable_predictions": stable,
        "total_predictions": total,
        "stability_rate": float(stable / total) if total else 0.0,
    }