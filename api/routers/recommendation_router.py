"""
Router for the Recommendation API.

Endpoints
---------
POST /api/recommend/
    Accept disease class, yield, and weather data; return suggestions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from api.services.weather import WeatherServiceError, get_weather
from modules.recommendation.engine import recommend, Recommendation
from modules.recommendation.predict import recommend_crop

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RecommendationInput(BaseModel):
    """Inputs for generating recommendations."""
    disease_class: str | None = Field(None, json_schema_extra={"example": "Tomato___Late_blight"})
    predicted_yield: float | None = Field(None, ge=0, json_schema_extra={"example": 3.5})
    temperature: float | None = Field(None, json_schema_extra={"example": 30.0})
    humidity: float | None = Field(None, ge=0, le=100, json_schema_extra={"example": 70.0})
    soil_type: str | None = Field(None, json_schema_extra={"example": "clay"})


class RecommendationOutput(BaseModel):
    fertilizer: str
    medicine: str
    notes: str


class CropRecommendationInput(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    location: str | None = Field(None, min_length=1)
    temperature: float | None = None
    humidity: float | None = Field(None, ge=0, le=100)
    rainfall: float | None = Field(None, ge=0)
    soil_moisture: float = Field(..., ge=0, le=100)
    pH: float = Field(..., ge=0, le=14)
    season: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=20)


class CropRecommendationOutput(BaseModel):
    recommendations: list[dict[str, str | float]]
    weather: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=RecommendationOutput)
async def get_recommendation(data: RecommendationInput):
    """Return fertiliser and medicine recommendations."""
    rec: Recommendation = recommend(
        disease_class=data.disease_class,
        predicted_yield=data.predicted_yield,
        temperature=data.temperature,
        humidity=data.humidity,
        soil_type=data.soil_type,
    )
    return RecommendationOutput(
        fertilizer=rec.fertilizer,
        medicine=rec.medicine,
        notes=rec.notes,
    )


@router.post("/crop", response_model=CropRecommendationOutput)
async def get_crop_recommendation(data: CropRecommendationInput):
    """Return ranked crop recommendations from the trained tabular model."""
    values = data.model_dump()
    weather = {
        "temperature": values.pop("temperature"),
        "humidity": values.pop("humidity"),
        "rainfall": values.pop("rainfall"),
    }
    if any(value is None for value in weather.values()):
        try:
            retrieved = get_weather(values.pop("location") or "")
        except WeatherServiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        weather = retrieved
    else:
        values.pop("location", None)
    result = recommend_crop(
        N=values["N"], P=values["P"], K=values["K"],
        temperature=weather["temperature"], humidity=weather["humidity"],
        rainfall=weather["rainfall"], soil_moisture=values["soil_moisture"],
        pH=values["pH"], season=values["season"],
        top_k=values["top_k"],
    )
    return {**result, "weather": weather}
