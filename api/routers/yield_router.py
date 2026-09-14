"""
Router for the Crop Yield Prediction API.

Endpoints
---------
POST /api/yield/predict
    Accept soil, weather, and crop features; return predicted yield.
POST /api/yield/train
    Trigger model (re-)training and return comparison metrics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from api.services.weather import WeatherServiceError, get_weather
from fastapi import APIRouter, HTTPException

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class YieldInput(BaseModel):
    """Input features for yield prediction."""
    soil_type: str = Field(..., json_schema_extra={"example": "clay"})
    location: str | None = Field(None, min_length=1)
    rainfall_mm: float | None = Field(None, ge=0, json_schema_extra={"example": 200.0})
    temperature_c: float | None = Field(None, json_schema_extra={"example": 28.5})
    humidity_pct: float | None = Field(None, ge=0, le=100, json_schema_extra={"example": 65.0})
    area_hectares: float = Field(..., gt=0, json_schema_extra={"example": 2.5})
    crop_type: str = Field(..., json_schema_extra={"example": "rice"})
    season: str = Field(..., json_schema_extra={"example": "kharif"})


class YieldOutput(BaseModel):
    predicted_yield: float
    unit: str = "tons/hectare"
    weather: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=YieldOutput)
async def predict_yield(data: YieldInput):
    """Predict crop yield from the provided features."""
    try:
        from modules.yield_prediction.predict import predict
        values = data.model_dump()
        weather = {
            "temperature": values.pop("temperature_c"),
            "humidity": values.pop("humidity_pct"),
            "rainfall": values.pop("rainfall_mm"),
        }
        if any(value is None for value in weather.values()):
            weather = get_weather(values.pop("location") or "")
        else:
            values.pop("location", None)
        values.update({
            "temperature_c": weather["temperature"],
            "humidity_pct": weather["humidity"],
            "rainfall_mm": weather["rainfall"],
        })
        result = predict(values)
        return YieldOutput(predicted_yield=round(result, 4), weather=weather)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Yield model not trained yet. Call POST /api/yield/train first.",
        )
    except WeatherServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/train")
async def train_yield_model():
    """Train (or retrain) the yield prediction model and return metrics."""
    try:
        from modules.yield_prediction.train import train_and_select
        summary = train_and_select()
        return summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
