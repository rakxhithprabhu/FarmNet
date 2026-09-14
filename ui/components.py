"""Reusable Streamlit presentation helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st


def show_recommendations(result: dict[str, Any]) -> None:
    recommendations = result.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        st.error("The recommendation service returned no crop recommendations.")
        return
    st.subheader("Recommended Crops")
    for index, item in enumerate(recommendations[:3], start=1):
        crop = item.get("crop")
        confidence = item.get("confidence")
        if not isinstance(crop, str) or not isinstance(confidence, (int, float)):
            st.error("The recommendation service returned an invalid result.")
            return
        st.metric(f"{index}. {crop.title()}", f"{confidence:.1%}")


def show_yield_result(result: dict[str, Any]) -> None:
    prediction = result.get("predicted_yield")
    unit = result.get("unit", "tons/hectare")
    if not isinstance(prediction, (int, float)):
        st.error("The yield service returned an invalid result.")
        return
    st.subheader("Predicted Yield")
    st.metric("Estimated production", f"{prediction:.2f} {unit}")
    weather = result.get("weather")
    if isinstance(weather, dict):
        st.caption(
            "Automatically retrieved: "
            f"{weather.get('temperature', 0):.1f} °C, "
            f"{weather.get('humidity', 0):.0f}% humidity, "
            f"{weather.get('rainfall', 0):.1f} mm rainfall"
        )


def show_disease_result(result: dict[str, Any]) -> None:
    disease = result.get("disease_class")
    confidence = result.get("confidence")
    if not isinstance(disease, str) or not isinstance(confidence, (int, float)):
        st.error("The disease service returned an invalid result.")
        return
    st.subheader("Disease Detection Result")
    st.metric("Detected condition", disease.replace("___", " - ").replace("_", " "))
    st.caption(f"Confidence: {confidence:.1%}")