"""
Central configuration for the Smart Agriculture Platform.

All paths, hyperparameters, and feature definitions are kept here so that
every module imports from a single source of truth.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Dataset
YIELD_DATASET_PATH = DATA_DIR / "crop_yield.csv"

# Saved Models
YIELD_MODEL_PATH = MODEL_DIR / "yield_model.joblib"

DISEASE_MODEL_PATH = MODEL_DIR / "disease_model.pt"
DISEASE_CLASS_NAMES_PATH = MODEL_DIR / "disease_classes.json"

# ---------------------------------------------------------------------------
# Yield Prediction – Feature Schema
# ---------------------------------------------------------------------------

# Numerical Features
YIELD_NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "NDVI",
    "GNDVI",
    "NDWI",
    "SAVI",
    "soil_moisture",
    "temperature",
    "rainfall",
    "N",
    "P",
    "K",
    "pH",
    "humidity",
    "month",
    "day_of_year",

]

# Categorical Features
YIELD_CATEGORICAL_FEATURES = [
    "crop_type",
    "season",
]

# Date Column
YIELD_DATE_FEATURE = "date_of_image"

# Target Column
YIELD_TARGET = "yield"

# Optional Columns (not used for training)
YIELD_IGNORE_COLUMNS = [
    "field_id",
]

# ---------------------------------------------------------------------------
# Yield Prediction – Training Hyperparameters
# ---------------------------------------------------------------------------

YIELD_TEST_SIZE = 0.20
YIELD_RANDOM_STATE = 42
YIELD_CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": YIELD_RANDOM_STATE,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": YIELD_RANDOM_STATE,
    "objective": "reg:squarederror",
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "random_state": YIELD_RANDOM_STATE,
}

CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 8,
    "verbose": False,
    "random_seed": YIELD_RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Explainable AI
# ---------------------------------------------------------------------------

ENABLE_SHAP = True

# ---------------------------------------------------------------------------
# Disease Detection
# ---------------------------------------------------------------------------

DISEASE_IMAGE_SIZE = (224, 224)
DISEASE_BATCH_SIZE = 32
DISEASE_EPOCHS = 10
DISEASE_LEARNING_RATE = 1e-4
DISEASE_VALIDATION_SPLIT = 0.2
DISEASE_NUM_CLASSES: int | None = None

# ---------------------------------------------------------------------------
# Weather API (Future Use)
# ---------------------------------------------------------------------------

WEATHER_API_URL = os.getenv(
    "WEATHER_API_URL",
    "https://api.openweathermap.org/data/2.5/weather",
)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")