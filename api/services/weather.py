"""Retrieve and validate weather features used by the ML models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from config.settings import WEATHER_API_URL


# Load .env from the FarmNet project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


class WeatherServiceError(RuntimeError):
    """Raised when required weather data cannot be retrieved or validated."""


def get_weather(location: str, *, timeout: float = 8.0) -> dict[str, float]:
    """Return model-ready weather values for an OpenWeather location."""

    location = location.strip()

    if not location:
        raise WeatherServiceError(
            "A farm location is required to retrieve weather data."
        )

    if not WEATHER_API_KEY:
        raise WeatherServiceError(
            "WEATHER_API_KEY is missing. Check the .env file."
        )

    try:
        response = requests.get(
            WEATHER_API_URL,
            params={
                "q": location,
                "appid": WEATHER_API_KEY,
                "units": "metric",
            },
            timeout=timeout,
        )

        if response.status_code == 401:
            raise WeatherServiceError(
                "OpenWeather rejected the API key. "
                "Check that WEATHER_API_KEY in .env is valid and active."
            )

        if response.status_code == 404:
            raise WeatherServiceError(
                f"Location '{location}' was not found by OpenWeather."
            )

        if response.status_code == 429:
            raise WeatherServiceError(
                "OpenWeather API request limit has been reached."
            )

        if 500 <= response.status_code < 600:
            raise WeatherServiceError(
                "OpenWeather is currently unavailable. Please try again later."
            )

        response.raise_for_status()

        payload: Any = response.json()
        main = payload.get("main", {})

        temperature = float(main["temp"])
        humidity = float(main["humidity"])

        rain = payload.get("rain", {})

        rainfall = float(
            rain.get(
                "1h",
                rain.get("3h", 0.0),
            )
        )

    except WeatherServiceError:
        raise

    except requests.Timeout as exc:
        raise WeatherServiceError(
            "The weather service timed out. Please try again."
        ) from exc

    except requests.ConnectionError as exc:
        raise WeatherServiceError(
            "Unable to connect to OpenWeather. Check your internet connection."
        ) from exc

    except requests.RequestException as exc:
        raise WeatherServiceError(
            "Unable to retrieve weather data from OpenWeather."
        ) from exc

    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherServiceError(
            "OpenWeather returned an unexpected weather response."
        ) from exc

    if not -100 <= temperature <= 100:
        raise WeatherServiceError(
            "The weather service returned an invalid temperature."
        )

    if not 0 <= humidity <= 100:
        raise WeatherServiceError(
            "The weather service returned invalid humidity data."
        )

    if rainfall < 0:
        raise WeatherServiceError(
            "The weather service returned invalid rainfall data."
        )

    return {
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
    }