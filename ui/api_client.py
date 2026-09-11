"""HTTP client for the FarmNet FastAPI backend."""

from __future__ import annotations

from typing import Any

import requests


class FarmNetAPIError(RuntimeError):
    """A user-safe error raised when the backend request cannot be completed."""


def _post(base_url: str, endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise FarmNetAPIError(
            f"The FarmNet service at {url} took too long to respond."
        ) from exc
    except requests.ConnectionError as exc:
        raise FarmNetAPIError(
            f"Unable to connect to the FarmNet prediction service at {url}. "
            "Please start the backend with the documented command."
        ) from exc
    except requests.RequestException as exc:
        raise FarmNetAPIError(
            f"The FarmNet service request to {url} could not be completed."
        ) from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "The backend rejected the request.")
        except ValueError:
            detail = "The backend rejected the request."
        raise FarmNetAPIError(f"FarmNet returned HTTP {response.status_code}: {detail}")
    try:
        result = response.json()
    except ValueError as exc:
        raise FarmNetAPIError("The backend returned an invalid response.") from exc
    if not isinstance(result, dict):
        raise FarmNetAPIError("The backend returned an invalid response.")
    return result


def predict_yield(data: dict[str, Any], base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    """Call the existing yield prediction endpoint."""
    return _post(base_url, "/api/yield/predict", data, timeout)


def recommend_crop(data: dict[str, Any], base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    """Call the existing trained crop recommendation endpoint."""
    return _post(base_url, "/api/recommend/crop", data, timeout)