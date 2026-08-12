"""
FastAPI application entry-point for the Smart Agriculture Platform.

Mounts three routers:
    /api/yield       – Crop Yield Prediction
    /api/disease     – Crop Disease Detection
    /api/recommend   – Fertiliser & Medicine Recommendation
"""

from fastapi import FastAPI

from api.routers.yield_router import router as yield_router
from api.routers.disease_router import router as disease_router
from api.routers.recommendation_router import router as rec_router

app = FastAPI(
    title="Smart Agriculture Platform",
    description=(
        "An integrated ML-based platform for crop yield forecasting, "
        "disease classification, and intelligent fertiliser / medicine "
        "recommendation."
    ),
    version="1.0.0",
)

app.include_router(yield_router, prefix="/api/yield", tags=["Yield Prediction"])
app.include_router(disease_router, prefix="/api/disease", tags=["Disease Detection"])
app.include_router(rec_router, prefix="/api/recommend", tags=["Recommendation"])


@app.get("/", tags=["Health"])
async def root():
    """Health-check / welcome endpoint."""
    return {
        "service": "Smart Agriculture Platform",
        "version": "1.0.0",
        "endpoints": ["/api/yield", "/api/disease", "/api/recommend"],
    }
