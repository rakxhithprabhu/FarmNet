"""
Recommendation Engine for Fertiliser and Medicine suggestions.

This module uses a **rule-based** approach that combines:
* Predicted disease class (from disease detection)
* Predicted yield level (from yield prediction)
* Current weather conditions (temperature, humidity)

The rules are intentionally transparent so that domain experts can audit
and extend them without retraining a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Recommendation:
    """Container for a single recommendation."""
    fertilizer: str
    medicine: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Disease → medicine mapping
# ---------------------------------------------------------------------------
DISEASE_MEDICINE_MAP: dict[str, str] = {
    # Tomato diseases
    "Tomato___Late_blight": "Mancozeb 75% WP or Metalaxyl + Mancozeb",
    "Tomato___Early_blight": "Chlorothalonil or Copper Oxychloride",
    "Tomato___Bacterial_spot": "Copper hydroxide spray",
    "Tomato___Leaf_Mold": "Chlorothalonil fungicide",
    "Tomato___Septoria_leaf_spot": "Mancozeb or Chlorothalonil",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Abamectin or Spiromesifen",
    "Tomato___Target_Spot": "Azoxystrobin fungicide",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Imidacloprid (vector control)",
    "Tomato___Tomato_mosaic_virus": "No chemical cure; remove infected plants",
    "Tomato___healthy": "No treatment needed",
    # Potato diseases
    "Potato___Early_blight": "Mancozeb or Chlorothalonil",
    "Potato___Late_blight": "Metalaxyl + Mancozeb",
    "Potato___healthy": "No treatment needed",
    # Apple diseases
    "Apple___Apple_scab": "Captan or Myclobutanil",
    "Apple___Black_rot": "Captan + Thiophanate-methyl",
    "Apple___Cedar_apple_rust": "Myclobutanil fungicide",
    "Apple___healthy": "No treatment needed",
    # Corn diseases
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Azoxystrobin + Propiconazole",
    "Corn_(maize)___Common_rust_": "Mancozeb or Propiconazole",
    "Corn_(maize)___Northern_Leaf_Blight": "Azoxystrobin fungicide",
    "Corn_(maize)___healthy": "No treatment needed",
    # Grape diseases
    "Grape___Black_rot": "Mancozeb or Myclobutanil",
    "Grape___Esca_(Black_Measles)": "Sodium arsenite (where permitted)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Copper-based fungicide",
    "Grape___healthy": "No treatment needed",
}

# ---------------------------------------------------------------------------
# Yield-level thresholds (tons / hectare)
# ---------------------------------------------------------------------------
LOW_YIELD_THRESHOLD = 2.0
HIGH_YIELD_THRESHOLD = 5.0


def _classify_yield(yield_value: float) -> str:
    if yield_value < LOW_YIELD_THRESHOLD:
        return "low"
    elif yield_value < HIGH_YIELD_THRESHOLD:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Fertiliser recommendation rules
# ---------------------------------------------------------------------------

def _recommend_fertilizer(
    yield_level: str,
    temperature: float | None = None,
    humidity: float | None = None,
    soil_type: str | None = None,
) -> str:
    """Simple rule-based fertiliser recommendation."""
    if yield_level == "low":
        base = "NPK 20-20-20 (balanced) + organic compost"
    elif yield_level == "medium":
        base = "Urea (46-0-0) for nitrogen boost"
    else:
        base = "Maintenance dose: DAP 18-46-0"

    # Weather adjustments
    if temperature is not None and temperature > 35:
        base += " | Add potassium sulphate for heat stress"
    if humidity is not None and humidity > 80:
        base += " | Reduce nitrogen to avoid fungal growth"
    if soil_type and soil_type.lower() in ("sandy", "loamy sand"):
        base += " | Increase organic matter"

    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend(
    disease_class: str | None = None,
    predicted_yield: float | None = None,
    temperature: float | None = None,
    humidity: float | None = None,
    soil_type: str | None = None,
) -> Recommendation:
    """
    Generate fertiliser and medicine recommendations.

    Parameters
    ----------
    disease_class : str, optional
        Disease class name from the disease detection model.
    predicted_yield : float, optional
        Predicted crop yield in tons/hectare.
    temperature : float, optional
        Current temperature in °C.
    humidity : float, optional
        Current relative humidity (%).
    soil_type : str, optional
        Soil type string (e.g., "clay", "sandy").

    Returns
    -------
    Recommendation
    """
    # Medicine
    medicine = "No specific medicine recommended"
    if disease_class:
        medicine = DISEASE_MEDICINE_MAP.get(
            disease_class,
            f"Consult local agronomist for '{disease_class}'",
        )

    # Fertiliser
    yield_level = _classify_yield(predicted_yield) if predicted_yield is not None else "medium"
    fertilizer = _recommend_fertilizer(yield_level, temperature, humidity, soil_type)

    notes_parts: list[str] = []
    if predicted_yield is not None:
        notes_parts.append(f"Yield level: {yield_level} ({predicted_yield:.2f} t/ha)")
    if disease_class:
        notes_parts.append(f"Detected disease: {disease_class}")
    if temperature is not None:
        notes_parts.append(f"Temperature: {temperature}°C")
    if humidity is not None:
        notes_parts.append(f"Humidity: {humidity}%")

    return Recommendation(
        fertilizer=fertilizer,
        medicine=medicine,
        notes=" | ".join(notes_parts) if notes_parts else "General recommendation",
    )
