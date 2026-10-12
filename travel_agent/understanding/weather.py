"""Replaceable weather providers; no network calls at import time."""

import json
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from travel_agent.schemas import TripShoppingRequest, WeatherContext


def get_json(url):
    with urlopen(url, timeout=15) as response:
        return json.load(response)


class FixtureWeather:
    """Explicit synthetic/historical fixture with exact request scope."""

    def __init__(self, payload):
        self.context = WeatherContext.model_validate(payload)
        if self.context.data_type == "forecast":
            raise ValueError("Use the live provider for forecasts, not user-supplied fixtures")

    def fetch(self, request):
        weather = self.context
        if (
            weather.destination != request.destination
            or weather.start_date != request.start_date
            or weather.end_date != request.end_date
        ):
            return WeatherContext(reason="Fixture location/date scope does not match the request")
        return weather.model_copy(deep=True)


class UnavailableWeather:
    def fetch(self, request):
        return WeatherContext(reason="No weather provider selected; no weather is assumed")


class OpenMeteoWeather:
    """Geocode a city, or accept coordinates explicitly confirmed by the caller.

    Ambiguous search results return candidates rather than picking a city silently.
    Forecast window is checked again against actual dates returned by the API.
    """

    def __init__(self, location=None, transport=get_json, today=None):
        self.location = location
        self.transport = transport
        self.today = today or datetime.now(ZoneInfo("Asia/Singapore")).date()

    def fetch(self, request: TripShoppingRequest):
        if not request.start_date or not request.end_date or not request.destination:
            return WeatherContext(reason="Destination and exact dates are required")
        if request.start_date < self.today or request.end_date > self.today + timedelta(days=15):
            return WeatherContext(
                status="out_of_range",
                reason=(
                    "Dates are outside the forecast window; no historical fallback is configured"
                ),
            )
        try:
            location = self.location
            if location is not None and location.get("destination") != request.destination:
                return WeatherContext(
                    status="needs_location",
                    reason=("Confirmed coordinates must be bound to the current destination"),
                )
            if location is None:
                query = urlencode({"name": request.destination, "count": 5, "language": "en"})
                candidates = self.transport(
                    "https://geocoding-api.open-meteo.com/v1/search?" + query
                ).get("results", [])
                if len(candidates) != 1:
                    return WeatherContext(
                        status="needs_location",
                        location_candidates=[
                            {
                                key: c.get(key)
                                for key in (
                                    "id",
                                    "name",
                                    "country",
                                    "admin1",
                                    "latitude",
                                    "longitude",
                                    "timezone",
                                )
                            }
                            for c in candidates
                        ],
                        reason="Confirm the intended city and coordinates",
                    )
                location = candidates[0]
            lat, lon = float(location["latitude"]), float(location["longitude"])
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError("Invalid coordinates")
            query = urlencode(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "timezone": "auto",
                    "start_date": request.start_date.isoformat(),
                    "end_date": request.end_date.isoformat(),
                    "daily": "temperature_2m_min,temperature_2m_max,"
                    "precipitation_probability_max,snowfall_sum",
                }
            )
            source = "https://api.open-meteo.com/v1/forecast?" + query
            payload = self.transport(source)
            daily = payload["daily"]
            units = payload["daily_units"]
            expected = {
                "temperature_2m_min": "°C",
                "temperature_2m_max": "°C",
                "precipitation_probability_max": "%",
                "snowfall_sum": "cm",
            }
            if any(units.get(key) != unit for key, unit in expected.items()):
                raise ValueError("Unexpected weather units")
            dates = [date.fromisoformat(d) for d in daily["time"]]
            needed = [
                request.start_date + timedelta(days=i)
                for i in range((request.end_date - request.start_date).days + 1)
            ]
            if dates != needed:
                raise ValueError("Incomplete weather date coverage")
            for key in expected:
                values = daily[key]
                if len(values) != len(needed) or any(v is None for v in values):
                    raise ValueError("Incomplete weather attributes")
            probabilities = daily["precipitation_probability_max"]
            if any(not 0 <= p <= 100 for p in probabilities):
                raise ValueError("Invalid precipitation probability")
            return WeatherContext(
                status="ok",
                data_type="forecast",
                destination=request.destination,
                start_date=request.start_date,
                end_date=request.end_date,
                source=source,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                latitude=lat,
                longitude=lon,
                location_label=location.get("name", request.destination),
                temp_min_celsius=min(daily["temperature_2m_min"]),
                temp_max_celsius=max(daily["temperature_2m_max"]),
                precipitation_probability=max(probabilities) / 100,
                snow=any(s > 0 for s in daily["snowfall_sum"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return WeatherContext(reason=f"Weather unavailable: {type(exc).__name__}")
