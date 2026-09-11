"""Random Forest model construction for crop recommendation."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from config.settings import (
    RECOMMENDATION_MAX_DEPTH,
    RECOMMENDATION_MIN_SAMPLES_LEAF,
    RECOMMENDATION_MIN_SAMPLES_SPLIT,
    RECOMMENDATION_N_ESTIMATORS,
    RECOMMENDATION_RANDOM_STATE,
)


def create_recommendation_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RECOMMENDATION_N_ESTIMATORS,
        max_depth=RECOMMENDATION_MAX_DEPTH,
        random_state=RECOMMENDATION_RANDOM_STATE,
        min_samples_split=RECOMMENDATION_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RECOMMENDATION_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        n_jobs=-1,
    )