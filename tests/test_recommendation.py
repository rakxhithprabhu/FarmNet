"""
Tests for the Recommendation Engine.

Covers:
* Rule-based fertiliser selection
* Disease → medicine mapping
* Edge cases (missing inputs)
"""

from __future__ import annotations

import pytest

from modules.recommendation.engine import (
    Recommendation,
    recommend,
    _classify_yield,
)


class TestClassifyYield:
    def test_low(self):
        assert _classify_yield(1.0) == "low"

    def test_medium(self):
        assert _classify_yield(3.5) == "medium"

    def test_high(self):
        assert _classify_yield(6.0) == "high"


class TestRecommend:
    def test_known_disease(self):
        rec = recommend(disease_class="Tomato___Late_blight")
        assert isinstance(rec, Recommendation)
        assert "Mancozeb" in rec.medicine or "Metalaxyl" in rec.medicine

    def test_unknown_disease_fallback(self):
        rec = recommend(disease_class="Unknown___Disease")
        assert "agronomist" in rec.medicine.lower()

    def test_no_disease(self):
        rec = recommend(predicted_yield=4.0)
        assert rec.medicine == "No specific medicine recommended"

    def test_low_yield_fertilizer(self):
        rec = recommend(predicted_yield=1.0)
        assert "NPK" in rec.fertilizer

    def test_high_temp_adjustment(self):
        rec = recommend(predicted_yield=3.0, temperature=40.0)
        assert "potassium" in rec.fertilizer.lower()

    def test_high_humidity_adjustment(self):
        rec = recommend(predicted_yield=3.0, humidity=85.0)
        assert "nitrogen" in rec.fertilizer.lower()

    def test_sandy_soil_adjustment(self):
        rec = recommend(predicted_yield=3.0, soil_type="sandy")
        assert "organic" in rec.fertilizer.lower()

    def test_all_inputs(self):
        rec = recommend(
            disease_class="Potato___Late_blight",
            predicted_yield=1.5,
            temperature=36.0,
            humidity=90.0,
            soil_type="sandy",
        )
        assert isinstance(rec, Recommendation)
        assert rec.fertilizer
        assert rec.medicine
        assert rec.notes

    def test_no_inputs(self):
        rec = recommend()
        assert isinstance(rec, Recommendation)
        assert rec.notes == "General recommendation"
