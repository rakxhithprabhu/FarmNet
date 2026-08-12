"""
Training script for the Crop Yield Prediction module.

Compares Random Forest, XGBoost, and Linear Regression.
Selects the best model based on cross-validated RMSE and persists it.
"""

from __future__ import annotations

import json
import logging
import importlib
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap


from pathlib import Path
from typing import Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score



from sklearn.model_selection import RandomizedSearchCV

from config.settings import (
    YIELD_CV_FOLDS,
    YIELD_MODEL_PATH,
    YIELD_RANDOM_STATE,
)
from modules.yield_prediction.preprocessing import (
    load_data,
    split_data,
)

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

try:
    LGBMRegressor = importlib.import_module("lightgbm").LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    LGBMRegressor = None
    _HAS_LGBM = False






def _get_candidate_models() -> dict[str, Any]:
    """Return a mapping of model name → unfitted estimator."""
    models: dict[str, Any] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=YIELD_RANDOM_STATE,
        ),
        "LinearRegression": LinearRegression(),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=YIELD_RANDOM_STATE,
        )

    if _HAS_LGBM and LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=YIELD_RANDOM_STATE,
        )
  
    return models


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Return RMSE, MAE, and R² for a fitted model on the test set."""
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }

# chatgpt 1-8
def tune_model(name: str, model: Any, X_train, y_train):

    if name == "RandomForest":

        params = {
            "n_estimators": [100,200,300,500],
            "max_depth": [8,10,12,15,None],
            "min_samples_split": [2,5,10],
            "min_samples_leaf": [1,2,4],
        }

    elif name == "XGBoost":

        params = {
            "n_estimators": [100,200,300],
            "max_depth": [4,6,8],
            "learning_rate": [0.01,0.05,0.1],
        }

    elif name == "LightGBM":

        params = {
            "n_estimators": [100,200,300],
            "max_depth": [4,6,8,-1],
            "learning_rate": [0.01,0.05,0.1],
        }

    else:
        return model

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=10,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=YIELD_RANDOM_STATE,
    )

    search.fit(X_train, y_train)

    logger.info("%s Best Parameters: %s",
                name,
                search.best_params_)

    return search.best_estimator_




def train_and_select(
    csv_path: str | None = None,
    cv_folds: int = YIELD_CV_FOLDS,
    save: bool = True,
) -> dict[str, Any]:
    """
    Full training pipeline:
    1. Load & preprocess data
    2. Train all candidate models
    3. Cross-validate
    4. Select best by RMSE
    5. Persist best model + preprocessor

    Returns a summary dict with per-model metrics and the best model name.
    """
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test, preprocessor = split_data(df)

    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in _get_candidate_models().items():
        logger.info("Training %s …", name)
        #gpt 1-8

        model = tune_model(
        name,
        model,
        X_train,
        y_train,
        )

        tune_model.fit(X_train, y_train)


        fitted_models[name] = model

        # Cross-validation (negative MSE → RMSE)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring="neg_mean_squared_error",
        )
        cv_rmse = float(np.mean(np.sqrt(-cv_scores)))

        test_metrics = evaluate_model(model, X_test, y_test)
        test_metrics["cv_rmse"] = cv_rmse
        results[name] = test_metrics
        logger.info("%s  →  %s", name, test_metrics)

    # Select best model (lowest test RMSE)
    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_model = fitted_models[best_name]

    # Feature importance (tree-based models only)
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = best_model.feature_importances_.tolist()

    if save:
        YIELD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": best_model, "preprocessor": preprocessor},
            YIELD_MODEL_PATH,
        )
        logger.info("Saved best model (%s) to %s", best_name, YIELD_MODEL_PATH)

    return {
        "best_model": best_name,
        "results": results,
        "feature_importance": feature_importance,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = train_and_select()
    print(json.dumps(summary, indent=2))
