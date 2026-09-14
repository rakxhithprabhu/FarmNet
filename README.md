# Smart Agriculture Platform

An **Integrated Machine Learning-Based Smart Agriculture Platform** that provides:

| Module | Purpose | Technique |
|---|---|---|
| **Yield Prediction** | Forecast crop yield (tons/hectare) | RF, XGBoost, Linear Regression |
| **Disease Detection** | Classify crop leaf diseases | CNN (ResNet-18 transfer learning) |
| **Recommendation** | Rank suitable crops from farm and weather inputs | Random Forest pipeline |

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


### 1. Create and Activate the Virtual Environment

PowerShell:

```powershell
cd FarmNet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
## Quick Start

Run all commands from the repository root (`FarmNet`). On Windows, use the project
virtual environment explicitly so the interpreter and installed packages match:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is unavailable, prefix commands with `.\.venv\Scripts\python.exe`.

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Generate Synthetic Dataset (for development)

```bash
python -m data.generate_dataset
```

### 4. Train the Yield Prediction Model

```bash
python -m modules.yield_prediction.train
```

### 5. Train the Disease Detection Model

Requires the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset):

```bash
python -m modules.disease_detection.train /path/to/plantvillage
```

### 6. Configure Weather Retrieval

Recommendation and yield prediction retrieve weather data on the backend. Set the
OpenWeather API key before starting the backend:

PowerShell:

```powershell
$env:WEATHER_API_KEY = "your-openweather-api-key"
```

Command Prompt:

```cmd
set WEATHER_API_KEY=your-openweather-api-key
```

Do not put the API key in Streamlit code or commit it to the repository.

### 7. Start the API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then visit **http://localhost:8000/docs** for the interactive Swagger UI.

Keep this terminal running.

### 8. Start the Streamlit UI

Open a second terminal, navigate to the repository root, and activate the same
virtual environment:

PowerShell:

```powershell
cd S:\Project26\Model0.1\App5\FarmNet
.\.venv\Scripts\Activate.ps1
streamlit run ui/app.py
```

If the environment is already activated, run only:

```powershell
streamlit run ui/app.py
```

Streamlit opens the UI at **http://localhost:8501**. The UI uses
`http://127.0.0.1:8000` for the backend by default. To use another backend URL:

```powershell
$env:FARMNET_API_URL = "http://127.0.0.1:8000"
streamlit run ui/app.py
```

In the UI:

1. Choose **Disease Detection**, **Crop Recommendation**, or **Crop Yield Prediction** from the sidebar.
2. Enter the farmer inputs shown by the selected page.
3. Enter the farm location for recommendation or yield prediction so the backend can retrieve weather.
4. Upload a crop leaf image for disease detection.
5. Submit the form and view the prediction result.

### 9. Run Tests

```bash
python -m pytest -q
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

## Input Architecture

The UI sends farmer values and a farm location to the backend. The backend retrieves
weather, validates both sources, merges them, and calls the existing model.

### Crop Recommendation

| Feature | Source | Validation |
|---|---|---|
| `N`, `P`, `K` | Farmer | Non-negative numeric |
| `pH` | Farmer | 0 to 14 |
| `season`, `soil_moisture` | Farmer | Non-empty season; moisture 0 to 100 |
| `temperature`, `humidity`, `rainfall` | Weather API | Validated numeric weather response |

### Yield Prediction

| Feature | Source | Validation |
|---|---|---|
| `soil_type`, `crop_type`, `season` | Farmer | Non-empty text |
| `area_hectares` | Farmer | Greater than 0 |
| `temperature_c`, `humidity_pct`, `rainfall_mm` | Weather API | Validated numeric weather response |

### Disease Detection

| Input | Source | Validation |
|---|---|---|
| Crop leaf image | User upload | JPEG, PNG, WEBP, or GIF |

### Yield Prediction
- **Random Forest** – robust to outliers, handles non-linear relationships, and provides feature importance.
- **XGBoost** – gradient boosting often achieves the lowest error on tabular data.
- **Linear Regression** – serves as a simple baseline for comparison.
- The best model is selected automatically based on test RMSE.

### Disease Detection
- **ResNet-18 with transfer learning** – pre-trained on ImageNet, only the final FC layer is fine-tuned. This keeps training fast and accurate even with limited data.
- Data augmentation (flips, rotations, colour jitter) helps prevent over-fitting.

### Recommendation Engine
- The crop recommendation endpoint uses the persisted Random Forest pipeline.
- The separate `/api/recommend/` endpoint remains a transparent rule-based fertiliser and medicine recommender.

---

## Deployment Recommendations

1. **Containerise** with Docker (one container per service or a single multi-stage image).
2. Serve the FastAPI app behind **Gunicorn + Uvicorn workers** for production.
3. Store trained models in **cloud object storage** (S3 / GCS) and load at startup.
4. Add **authentication** (OAuth2 / API keys) before exposing endpoints publicly.
5. Use **CI/CD** (GitHub Actions) to run tests and rebuild Docker images on push.
