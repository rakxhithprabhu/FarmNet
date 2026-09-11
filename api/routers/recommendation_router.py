"""
Router for the Recommendation API.

Endpoints
---------
POST /api/recommend/
    Accept disease class, yield, and weather data; return suggestions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter

from modules.recommendation.engine import recommend, recommend_crop, Recommendation

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
    soil: str = Field(..., min_length=1)
    season: str = Field(..., min_length=1)
    water_source: str = Field(..., min_length=1)
    soil_ph: float = Field(..., ge=0, le=14)
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    top_k: int = Field(3, ge=1, le=20)


class CropRecommendationOutput(BaseModel):
    recommendations: list[dict[str, str | float]]


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
    return recommend_crop(
        soil=values["soil"],
        season=values["season"],
        water_source=values["water_source"],
        soil_ph=values["soil_ph"],
        temperature=values["temperature"],
        humidity=values["humidity"],
        nitrogen=values["N"],
        phosphorus=values["P"],
        potassium=values["K"],
        top_k=values["top_k"],
    )
