"""
Generate a synthetic crop-yield dataset for development and testing.

Run directly:
    python -m App1.data.generate_dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import YIELD_DATASET_PATH


SOIL_TYPES = ["clay", "sandy", "loamy", "silt", "peat"]
CROP_TYPES = ["rice", "wheat", "maize", "sugarcane", "cotton", "potato"]
SEASONS = ["kharif", "rabi", "zaid"]


def generate(num_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic crop-yield CSV and return the DataFrame."""
    rng = np.random.default_rng(seed)

    soil = rng.choice(SOIL_TYPES, size=num_rows)
    crop = rng.choice(CROP_TYPES, size=num_rows)
    season = rng.choice(SEASONS, size=num_rows)
    rainfall = rng.uniform(50, 400, size=num_rows).round(1)
    temperature = rng.uniform(15, 42, size=num_rows).round(1)
    humidity = rng.uniform(30, 95, size=num_rows).round(1)
    area = rng.uniform(0.5, 20, size=num_rows).round(2)

    # Synthetic yield = f(rainfall, temp, humidity) + noise
    base_yield = (
        0.005 * rainfall
        + 0.02 * (35 - np.abs(temperature - 28))
        + 0.01 * humidity
        - 0.02 * area
    )
    noise = rng.normal(0, 0.3, size=num_rows)
    yield_val = np.clip(base_yield + noise, 0.5, 12).round(2)

    df = pd.DataFrame(
        {
            "soil_type": soil,
            "rainfall_mm": rainfall,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "area_hectares": area,
            "crop_type": crop,
            "season": season,
            "yield_tons_per_hectare": yield_val,
        }
    )

    YIELD_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(YIELD_DATASET_PATH, index=False)
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Generated {len(df)} rows → {YIELD_DATASET_PATH}")
    print(df.head())
