# FarmNet UI

This Streamlit UI calls the existing FastAPI backend for yield prediction and crop recommendation. It does not contain or duplicate model logic.

## Start the backend

From the project root:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Start the UI

In a second terminal from the project root:

```bash
streamlit run ui/app.py
```

The UI uses `http://127.0.0.1:8000` by default. Set `FARMNET_API_URL` to use another backend URL.