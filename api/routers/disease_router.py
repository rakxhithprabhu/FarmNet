"""
Router for the Crop Disease Detection API.

Endpoints
---------
POST /api/disease/predict
    Accept an image upload and return the predicted disease class.
POST /api/disease/train
    Trigger model training on a given dataset path.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class DiseaseOutput(BaseModel):
    disease_class: str
    confidence: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=DiseaseOutput)
async def predict_disease(file: UploadFile = File(...)):
    """Classify a crop leaf image and return the disease class."""
    try:
        from modules.disease_detection.predict import predict
        contents = await file.read()
        result = predict(contents)
        return DiseaseOutput(
            disease_class=result["class"],
            confidence=result["confidence"],
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Disease model not trained yet. Call POST /api/disease/train first.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/train")
async def train_disease_model(dataset_root: str):
    """
    Train the disease detection model.

    Parameters
    ----------
    dataset_root : str
        Server-side path to PlantVillage dataset root.
    """
    try:
        from modules.disease_detection.train import train
        summary = train(dataset_root)
        return summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
