"""SHAP explanations for the persisted crop recommendation Pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from modules.recommendation.preprocessing import FEATURES, build_feature_frame


def _transformed_parts(pipeline, frame: pd.DataFrame) -> tuple[Any, np.ndarray, list[str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(frame)
    names = list(preprocessor.get_feature_names_out())
    return classifier, np.asarray(transformed), names


def _class_shap_values(explainer: shap.TreeExplainer, transformed: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        return np.asarray(values)
    values = np.asarray(values)
    if values.ndim == 3:
        return np.moveaxis(values, 2, 0)
    return values[np.newaxis, ...]


def _feature_groups(names: list[str]) -> list[str]:
    groups = []
    for name in names:
        groups.append(next((feature for feature in FEATURES if name == feature or name.startswith(f"{feature}_")), name))
    return groups


def _aggregate_importance(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    importance = np.mean(np.abs(values), axis=(0, 1))
    grouped: dict[str, float] = {feature: 0.0 for feature in FEATURES}
    for name, value in zip(_feature_groups(names), importance):
        grouped[name] = grouped.get(name, 0.0) + float(value)
    return pd.DataFrame({"feature": list(grouped), "mean_abs_shap": list(grouped.values())}).sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)


def generate_shap_reports(
    pipeline,
    frame: pd.DataFrame,
    output_dir: str | Path = "reports/recommendation",
    max_samples: int = 500,
) -> dict[str, Any]:
    """Generate sampled global and local SHAP reports for a trained Pipeline."""
    if frame.empty:
        raise ValueError("At least one input row is required for SHAP reports")
    sample = frame.sample(min(max_samples, len(frame)), random_state=42)
    classifier, transformed, names = _transformed_parts(pipeline, sample)
    explainer = shap.TreeExplainer(classifier)
    values = _class_shap_values(explainer, transformed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    importance = _aggregate_importance(values, names)
    importance.to_csv(output_path / "shap_feature_importance.csv", index=False)

    plot_values = values[classifier.classes_.tolist().index(classifier.predict(transformed)[0])]
    shap.summary_plot(plot_values, transformed, feature_names=names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path / "shap_summary.png", dpi=150)
    plt.close()

    local = explain_transformed_prediction(pipeline, sample.iloc[[0]])
    local_frame = pd.DataFrame(local["features"])
    local_frame.to_csv(output_path / "shap_local_example.csv", index=False)
    return {"importance": importance, "local": local, "sample_size": len(sample)}


def explain_transformed_prediction(pipeline, frame: pd.DataFrame) -> dict[str, Any]:
    """Return a user-facing local explanation for one already-shaped input row."""
    if len(frame) != 1:
        raise ValueError("Exactly one input row is required for a local explanation")
    classifier, transformed, names = _transformed_parts(pipeline, frame)
    explainer = shap.TreeExplainer(classifier)
    values = _class_shap_values(explainer, transformed)
    predicted_index = int(np.argmax(classifier.predict_proba(transformed)[0]))
    local_values = values[predicted_index, 0]
    grouped_impacts: dict[str, float] = {}
    for name, impact in zip(names, local_values):
        group = _feature_groups([name])[0]
        grouped_impacts[group] = grouped_impacts.get(group, 0.0) + float(impact)
    features = []
    for feature in FEATURES:
        impact = grouped_impacts.get(feature, 0.0)
        value = frame.iloc[0][feature]
        features.append({
            "feature": feature,
            "value": value,
            "impact": round(impact, 6),
            "direction": "positive" if impact >= 0 else "negative",
        })
    return {
        "predicted_crop": str(classifier.classes_[predicted_index]),
        "features": features,
    }


def explain_prediction(pipeline, input_data: dict[str, Any]) -> dict[str, Any]:
    """Validate public recommendation inputs and explain one prediction."""
    frame = build_feature_frame(input_data, FEATURES)
    return explain_transformed_prediction(pipeline, frame)