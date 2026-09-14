# FarmNet UI

This Streamlit UI calls the existing FastAPI backend for disease detection, yield prediction, and crop recommendation. It does not contain or duplicate model logic.

Recommendation and yield forms separate farmer-entered values from automatically
retrieved weather. Set `WEATHER_API_KEY` in the backend environment before using
location-based predictions; the backend never substitutes fake weather values.

## Start the backend

From the project root:

```bash
.venv/Scripts/python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Start the UI

In a second terminal from the project root:

```bash
streamlit run ui/app.py
```

The UI uses `http://127.0.0.1:8000` by default. Set `FARMNET_API_URL` to use another backend URL.

Pages:

- Disease Detection: upload a JPEG, PNG, WEBP, or GIF crop leaf image.
- Crop Recommendation: enter nutrients, pH, season, and soil moisture; weather is retrieved by location.
- Crop Yield Prediction: enter soil, crop, season, and area; weather is retrieved by location.