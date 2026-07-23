# 🌾 Smart Agriculture Platform

An **Integrated Machine Learning-Based Smart Agriculture Platform** that provides:

| Module | Purpose | Technique |
|---|---|---|
| **Yield Prediction** | Forecast crop yield (tons/hectare) | RF, XGBoost, Linear Regression |
| **Disease Detection** | Classify crop leaf diseases | CNN (ResNet-18 transfer learning) |
| **Recommendation** | Suggest fertiliser & medicine | Rule-based engine |

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

### 1. Install Dependencies

```bash
pip install -r smart_agriculture/requirements.txt
```

### 2. Generate Synthetic Dataset (for development)

```bash
python -m smart_agriculture.data.generate_dataset
```

### 3. Train the Yield Prediction Model

```bash
python -m smart_agriculture.modules.yield_prediction.train
```

### 4. Train the Disease Detection Model

Requires the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset):

```bash
python -m smart_agriculture.modules.disease_detection.train /path/to/plantvillage
```

### 5. Start the API Server

```bash
uvicorn smart_agriculture.api.main:app --reload --host 0.0.0.0 --port 8000
```

Then visit **http://localhost:8000/docs** for the interactive Swagger UI.

### 6. Run Tests

```bash
pytest smart_agriculture/tests/ -v
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

### Recommendation Engine
- A **transparent, rule-based** system maps disease classes to medicines and yield levels to fertiliser suggestions.
- Weather adjustments (temperature, humidity) modify recommendations in real time.
- Easy for domain experts to audit and extend without retraining.

---

## Deployment Recommendations

1. **Containerise** with Docker (one container per service or a single multi-stage image).
2. Serve the FastAPI app behind **Gunicorn + Uvicorn workers** for production.
3. Store trained models in **cloud object storage** (S3 / GCS) and load at startup.
4. Add **authentication** (OAuth2 / API keys) before exposing endpoints publicly.
5. Use **CI/CD** (GitHub Actions) to run tests and rebuild Docker images on push.
