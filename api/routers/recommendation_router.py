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

from modules.recommendation.engine import recommend, Recommendation

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
