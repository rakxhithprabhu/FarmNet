"""Random Forest model construction for crop recommendation."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from config.settings import (
    RECOMMENDATION_MAX_DEPTH,
    RECOMMENDATION_MIN_SAMPLES_LEAF,
    RECOMMENDATION_MIN_SAMPLES_SPLIT,
    RECOMMENDATION_N_ESTIMATORS,
    RECOMMENDATION_RANDOM_STATE,
)


def create_recommendation_model(**overrides) -> RandomForestClassifier:
    parameters = {
        "n_estimators": RECOMMENDATION_N_ESTIMATORS,
        "max_depth": RECOMMENDATION_MAX_DEPTH,
        "random_state": RECOMMENDATION_RANDOM_STATE,
        "min_samples_split": RECOMMENDATION_MIN_SAMPLES_SPLIT,
        "min_samples_leaf": RECOMMENDATION_MIN_SAMPLES_LEAF,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    parameters.update(overrides)
    return RandomForestClassifier(**parameters)


def create_recommendation_search(pipeline, *, n_iter: int = 20, cv: int = 3) -> RandomizedSearchCV:
    """Create a bounded macro-F1 search over the classifier in a Pipeline."""
    search_space = {
        "classifier__n_estimators": [200, 300, 500, 700],
        "classifier__max_depth": [None, 10, 20, 30, 40],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.5, 1.0],
        "classifier__bootstrap": [True, False],
    }
    return RandomizedSearchCV(
        pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=RECOMMENDATION_RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )