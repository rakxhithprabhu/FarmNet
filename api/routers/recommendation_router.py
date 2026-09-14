"""
Router for the Recommendation API.

Endpoints
---------
POST /api/recommend/
    Accept disease class, yield, and weather data; return suggestions.

POST /api/recommend/crop
    Return ranked crop recommendations from the trained tabular model.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.services.weather import WeatherServiceError, get_weather
from modules.recommendation.engine import recommend, Recommendation
from modules.recommendation.predict import recommend_crop


router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RecommendationInput(BaseModel):
    """Inputs for generating fertilizer and medicine recommendations."""

    disease_class: str | None = Field(
        None,
        json_schema_extra={"example": "Tomato___Late_blight"},
    )
    predicted_yield: float | None = Field(
        None,
        ge=0,
        json_schema_extra={"example": 3.5},
    )
    temperature: float | None = Field(
        None,
        json_schema_extra={"example": 30.0},
    )
    humidity: float | None = Field(
        None,
        ge=0,
        le=100,
        json_schema_extra={"example": 70.0},
    )
    soil_type: str | None = Field(
        None,
        json_schema_extra={"example": "clay"},
    )


class RecommendationOutput(BaseModel):
    fertilizer: str
    medicine: str
    notes: str


class CropRecommendationInput(BaseModel):
    """
    Inputs for crop recommendation.

    Farmer-entered inputs:
    - soil
    - season
    - water_source
    - soil_ph
    - N
    - P
    - K

    API-retrieved inputs:
    - temperature
    - humidity

    If temperature/humidity are not supplied by the client,
    they are retrieved automatically using the farm location.
    """

    # Farmer inputs
    soil: str = Field(..., min_length=1)
    season: str = Field(..., min_length=1)
    water_source: str = Field(..., min_length=1)

    soil_ph: float = Field(..., ge=0, le=14)

    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)

    # API / automatically retrieved inputs
    location: str | None = Field(None, min_length=1)
    temperature: float | None = None
    humidity: float | None = Field(None, ge=0, le=100)

    # Number of recommendations
    top_k: int = Field(3, ge=1, le=20)


class CropRecommendationOutput(BaseModel):
    recommendations: list[dict[str, str | float]]
    weather: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=RecommendationOutput)
async def get_recommendation(data: RecommendationInput):
    """Return fertilizer and medicine recommendations."""

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

    # -------------------------------------------------------
    # Retrieve weather automatically when it is not supplied
    # -------------------------------------------------------

    temperature = values.pop("temperature")
    humidity = values.pop("humidity")
    location = values.pop("location")

    weather: dict[str, float] | None = None

    if temperature is None or humidity is None:
        if not location:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Farm location is required when temperature "
                    "and humidity are not provided."
                ),
            )

        try:
            weather = get_weather(location)
        except WeatherServiceError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

        temperature = weather["temperature"]
        humidity = weather["humidity"]

    else:
        weather = {
            "temperature": temperature,
            "humidity": humidity,
        }

    # -------------------------------------------------------
    # Run the current recommendation model
    # -------------------------------------------------------

    result = recommend_crop(
        soil=values["soil"],
        season=values["season"],
        water_source=values["water_source"],
        soil_ph=values["soil_ph"],
        temperature=temperature,
        humidity=humidity,
        nitrogen=values["N"],
        phosphorus=values["P"],
        potassium=values["K"],
        top_k=values["top_k"],
    )

    return {
        **result,
        "weather": weather,
    }