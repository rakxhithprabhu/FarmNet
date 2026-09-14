"""FarmNet Streamlit dashboard."""

from __future__ import annotations

import os

import streamlit as st

try:
    from ui.api_client import FarmNetAPIError, predict_disease, predict_yield, recommend_crop
    from ui.components import show_disease_result, show_recommendations, show_yield_result
except ModuleNotFoundError:
    from api_client import FarmNetAPIError, predict_disease, predict_yield, recommend_crop
    from components import show_disease_result, show_recommendations, show_yield_result

st.set_page_config(page_title="FarmNet", page_icon="🌾", layout="wide")

API_URL = os.getenv("FARMNET_API_URL", "http://127.0.0.1:8000")


def _home() -> None:
    st.title("FarmNet")
    st.subheader("Smart Agriculture Platform")
    st.write("AI-powered agricultural decision support for crop yield and crop selection.")
    left, right = st.columns(2)
    with left:
        st.info("Use environmental, soil, and crop inputs to estimate yield.")
        if st.button("Crop Yield Prediction", use_container_width=True):
            st.session_state.page = "Crop Yield Prediction"
            st.rerun()
    with right:
        st.info("Rank suitable crops from current agricultural conditions.")
        if st.button("Crop Recommendation", use_container_width=True):
            st.session_state.page = "Crop Recommendation"
            st.rerun()


def _yield_page() -> None:
    st.title("Crop Yield Prediction")
    st.caption("Prediction is provided by the FarmNet FastAPI backend.")
    with st.form("yield_form"):
        st.subheader("Farmer Inputs")
        area = st.number_input("Area (hectares)", min_value=0.01, value=2.5)
        soil = st.selectbox("Soil type", ["clay", "loamy", "sandy", "black", "red"])
        crop = st.selectbox("Crop type", ["rice", "wheat", "maize", "cotton", "sugarcane"])
        season = st.selectbox("Season", ["kharif", "rabi", "summer", "winter"])
        st.subheader("Automatically Retrieved Data")
        location = st.text_input("Farm location", placeholder="City or town")
        submitted = st.form_submit_button("Predict Yield", type="primary")
    if submitted:
        if area <= 0:
            st.error("Area must be greater than zero.")
            return
        payload = {
            "soil_type": soil,
            "location": location,
            "area_hectares": area,
            "crop_type": crop,
            "season": season,
        }
        if not location.strip():
            st.error("Enter a farm location so weather data can be retrieved.")
            return
        try:
            with st.spinner("Getting automatic data and predicting yield..."):
                show_yield_result(predict_yield(payload, API_URL))
        except FarmNetAPIError as exc:
            st.error(str(exc))


def _recommendation_page() -> None:
    st.title("Crop Recommendation")
    st.caption("Ranked recommendations are provided by the trained FarmNet model.")
    with st.form("recommendation_form"):
        st.subheader("Farmer Inputs")
        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer", "Winter"])
        n, p, k, ph = st.columns(4)
        nitrogen = n.number_input("Nitrogen (N)", min_value=0.0, value=90.0)
        phosphorus = p.number_input("Phosphorus (P)", min_value=0.0, value=42.0)
        potassium = k.number_input("Potassium (K)", min_value=0.0, value=43.0)
        soil_ph = ph.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        soil_moisture = st.number_input("Soil moisture (%)", min_value=0.0, max_value=100.0, value=65.0)
        st.subheader("Automatically Retrieved Data")
        location = st.text_input("Farm location", placeholder="City or town")
        submitted = st.form_submit_button("Recommend Crops", type="primary")
    if submitted:
        payload = {
            "season": season, "soil_moisture": soil_moisture,
            "pH": soil_ph, "N": nitrogen, "P": phosphorus, "K": potassium,
            "location": location, "top_k": 3,
        }
        if not location.strip():
            st.error("Enter a farm location so weather data can be retrieved.")
            return
        try:
            with st.spinner("Getting automatic data and generating recommendations..."):
                result = recommend_crop(payload, API_URL)
            show_recommendations(result)
            weather = result.get("weather")
            if isinstance(weather, dict):
                st.caption(
                    "Automatically retrieved: "
                    f"{weather.get('temperature', 0):.1f} °C, "
                    f"{weather.get('humidity', 0):.0f}% humidity"
                )
        except FarmNetAPIError as exc:
            st.error(str(exc))


def main() -> None:
    st.sidebar.title("FarmNet")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Crop Yield Prediction", "Crop Recommendation", "Disease Detection"],
        index=["Home", "Crop Yield Prediction", "Crop Recommendation", "Disease Detection"].index(
            st.session_state.get("page", "Home")
        ),
    )
    st.session_state.page = page
    if page == "Home":
        _home()
    elif page == "Crop Yield Prediction":
        _yield_page()
    elif page == "Crop Recommendation":
        _recommendation_page()
    else:
        _disease_page()


def _disease_page() -> None:
    st.title("Disease Detection")
    st.caption("Upload a clear crop leaf image for analysis.")
    image = st.file_uploader("Crop leaf image", type=["jpg", "jpeg", "png", "webp"])
    if image is not None and st.button("Detect Disease", type="primary"):
        try:
            with st.spinner("Uploading image and running disease detection..."):
                show_disease_result(predict_disease(image.getvalue(), image.name, API_URL))
        except FarmNetAPIError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()