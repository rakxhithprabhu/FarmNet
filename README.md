# Smart Agriculture Platform

An **Integrated Machine Learning-Based Smart Agriculture Platform** that provides:

| Module | Purpose | Technique |
|---|---|---|
| **Yield Prediction** | Forecast crop yield (tons/hectare) | RF, XGBoost, Linear Regression |
| **Disease Detection** | Classify crop leaf diseases | CNN (ResNet-18 transfer learning) |
| **Recommendation** | Rank suitable crops from soil and weather inputs | Random Forest pipeline |

> **No IoT / embedded hardware required.** Weather data is fetched via API.

---

## Project Structure

```
smart_agriculture/
├── config/
│   └── settings.py              # Central configuration
├── data/
│   ├── generate_dataset.py      # Synthetic dataset generator
│   └── crop_yield.csv            # (generated)
├── models/                       # Persisted model artefacts
├── modules/
│   ├── yield_prediction/
│   │   ├── preprocessing.py      # Data cleaning & feature encoding
│   │   ├── train.py              # Model comparison & selection
│   │   └── predict.py            # Inference helper
│   ├── disease_detection/
│   │   ├── preprocessing.py      # Image transforms & augmentation
│   │   ├── train.py              # ResNet-18 fine-tuning
│   │   └── predict.py            # Image classification inference
│   └── recommendation/
│       └── engine.py             # Rule-based recommendation logic
├── api/
│   ├── main.py                   # FastAPI app entry-point
│   └── routers/
│       ├── yield_router.py       # /api/yield endpoints
│       ├── disease_router.py     # /api/disease endpoints
│       └── recommendation_router.py  # /api/recommend endpoints
├── tests/
│   ├── test_yield_prediction.py
│   ├── test_disease_detection.py
│   ├── test_recommendation.py
│   └── test_api.py
└── requirements.txt
```

---

## Quick Start

Run all commands from the repository root (`FarmNet`). On Windows, use the project
virtual environment explicitly so the interpreter and installed packages match:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is unavailable, prefix commands with `.\.venv\Scripts\python.exe`.

### 1. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Train the Crop Recommendation Model

```powershell
python -m modules.recommendation.train
```

Optional tuning, validation, and SHAP reports:

```powershell
python -m modules.recommendation.train --tune --enhanced-validation --reports
```

The command saves `models/crop_recommendation.joblib` and
`models/crop_recommendation_metadata.json`.

### 3. Start the API Server

```powershell
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for Swagger UI. The crop recommendation endpoint is
`POST /api/recommend/crop` and expects `soil`, `season`, `water_source`, `soil_ph`,
`N`, `P`, `K`, `temperature`, `humidity`, and optional `top_k`.

Example request:

```powershell
curl.exe -X POST http://localhost:8000/api/recommend/crop `
	-H "Content-Type: application/json" `
	-d '{"soil":"Loamy soil","season":"Kharif","water_source":"irrigated","soil_ph":6.5,"N":90,"P":40,"K":40,"temperature":25,"humidity":75,"top_k":3}'
```

### 4. Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/yield/predict` | Predict crop yield |
| `POST` | `/api/yield/train` | Train / retrain yield model |
| `POST` | `/api/disease/predict` | Classify leaf disease (image upload) |
| `POST` | `/api/disease/train` | Train disease model |
| `POST` | `/api/recommend/` | Get fertiliser & medicine recommendation |
| `POST` | `/api/recommend/crop` | Rank crops from soil and weather inputs |

---

## Model Design Decisions

### Yield Prediction
- **Random Forest** – robust to outliers, handles non-linear relationships, and provides feature importance.
- **XGBoost** – gradient boosting often achieves the lowest error on tabular data.
- **Linear Regression** – serves as a simple baseline for comparison.
- The best model is selected automatically based on test RMSE.

### Disease Detection
- **ResNet-18 with transfer learning** – pre-trained on ImageNet, only the final FC layer is fine-tuned. This keeps training fast and accurate even with limited data.
- Data augmentation (flips, rotations, colour jitter) helps prevent over-fitting.

### Crop Recommendation Model
- A preprocessing and Random Forest pipeline handles categorical and numerical features.
- The training command reports evaluation metrics and persists the model plus metadata.
- The separate `/api/recommend/` endpoint remains the transparent rule-based fertiliser and medicine recommender.

---

## Deployment Recommendations

1. **Containerise** with Docker (one container per service or a single multi-stage image).
2. Serve the FastAPI app behind **Gunicorn + Uvicorn workers** for production.
3. Store trained models in **cloud object storage** (S3 / GCS) and load at startup.
4. Add **authentication** (OAuth2 / API keys) before exposing endpoints publicly.
5. Use **CI/CD** (GitHub Actions) to run tests and rebuild Docker images on push.
