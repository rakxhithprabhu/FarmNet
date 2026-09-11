"""Explainability helpers for the crop recommendation model."""

from __future__ import annotations

import pandas as pd


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