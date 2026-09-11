from __future__ import annotations

import pytest
import requests

from ui.api_client import FarmNetAPIError, _post


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_api_client_posts_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return FakeResponse(payload={"predicted_yield": 4.2, "unit": "tons/hectare"})

    monkeypatch.setattr("ui.api_client.requests.post", fake_post)
    result = _post("http://localhost:8000/", "/api/yield/predict", {"crop_type": "rice"}, 5)
    assert result["predicted_yield"] == 4.2
    assert captured == {
        "url": "http://localhost:8000/api/yield/predict",
        "json": {"crop_type": "rice"},
        "timeout": 5,
    }


def test_api_client_handles_connection_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.ConnectionError

    monkeypatch.setattr("ui.api_client.requests.post", fake_post)
    with pytest.raises(FarmNetAPIError, match="Unable to connect"):
        _post("http://localhost:8000", "/api/recommend/crop", {}, 5)


def test_api_client_handles_invalid_response(monkeypatch):
    monkeypatch.setattr("ui.api_client.requests.post", lambda *args, **kwargs: FakeResponse(payload=[]))
    with pytest.raises(FarmNetAPIError, match="invalid response"):
        _post("http://localhost:8000", "/api/recommend/crop", {}, 5)