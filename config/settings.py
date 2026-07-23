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

YIELD_DATASET_PATH = DATA_DIR / "crop_yield.csv"
YIELD_MODEL_PATH = MODEL_DIR / "yield_model.joblib"

DISEASE_MODEL_PATH = MODEL_DIR / "disease_model.pt"
DISEASE_CLASS_NAMES_PATH = MODEL_DIR / "disease_classes.json"

# ---------------------------------------------------------------------------
# Yield Prediction – feature schema
# ---------------------------------------------------------------------------
YIELD_NUMERIC_FEATURES = [
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "area_hectares",
]
YIELD_CATEGORICAL_FEATURES = [
    "soil_type",
    "crop_type",
    "season",
]
YIELD_TARGET = "yield_tons_per_hectare"

# ---------------------------------------------------------------------------
# Yield Prediction – training hyper-parameters
# ---------------------------------------------------------------------------
YIELD_TEST_SIZE = 0.2
YIELD_CV_FOLDS = 5
YIELD_RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Disease Detection – training hyper-parameters
# ---------------------------------------------------------------------------
DISEASE_IMAGE_SIZE = (224, 224)
DISEASE_BATCH_SIZE = 32
DISEASE_EPOCHS = 10
DISEASE_LEARNING_RATE = 1e-4
DISEASE_VALIDATION_SPLIT = 0.2
DISEASE_NUM_CLASSES: int | None = None  # set dynamically during training

# ---------------------------------------------------------------------------
# Weather API (placeholder – no IoT)
# ---------------------------------------------------------------------------
WEATHER_API_URL = os.getenv(
    "WEATHER_API_URL",
    "https://api.openweathermap.org/data/2.5/weather",
)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
