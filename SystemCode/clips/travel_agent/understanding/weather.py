"""Replaceable weather providers; no network calls at import time."""

import json
import sqlite3
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
        self.today_override = today

    def resolve_location(self, request):
        from travel_agent.understanding.weather_locations import (
            WeatherLocationResolver,
            country_for,
        )

        country = country_for(request.destination)
        if (country and not country.get("city_state") and self.location is None) or (
            self.location is not None and self.location.get("scope", "city") != "city"
        ):
            return WeatherLocationResolver(self.transport).resolve(
                request.destination, self.location
            )
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
            if location.get("feature_code") and not location["feature_code"].startswith("PPL"):
                return WeatherLocationResolver.scope_question(request.destination)
        return location

    def fetch(self, request: TripShoppingRequest):
        if not request.start_date or not request.end_date or not request.destination:
            return WeatherContext(reason="Destination and exact dates are required")
        if self.today_override is not None and (
            request.start_date < self.today or request.end_date > self.today + timedelta(days=15)
        ):
            return WeatherContext(
                status="out_of_range",
                reason=(
                    "Dates are outside today through day 15; this forecast-only provider "
                    "does not substitute historical data"
                ),
            )
        source = None
        try:
            location = self.resolve_location(request)
            if isinstance(location, WeatherContext):
                return location
            lat, lon = float(location["latitude"]), float(location["longitude"])
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError("Invalid coordinates")
            today = (
                self.today_override
                or datetime.now(ZoneInfo(location.get("timezone") or "Asia/Singapore")).date()
            )
            if request.start_date < today or request.end_date > today + timedelta(days=15):
                return WeatherContext(
                    status="out_of_range",
                    reason="Outside forecast window for the resolved location",
                )
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
                location_basis=(
                    location.get("location_basis", "caller_confirmed_coordinates")
                    if self.location is not None
                    else "provider_geocoded_city"
                ),
                temp_min_celsius=min(daily["temperature_2m_min"]),
                temp_max_celsius=max(daily["temperature_2m_max"]),
                precipitation_probability=max(probabilities) / 100,
                snow=any(s > 0 for s in daily["snowfall_sum"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return WeatherContext(
                source=source, reason=f"Weather unavailable: {type(exc).__name__}"
            )


class PlanningWeather:
    """Use one truthful data type per trip; API failures never silently become history."""

    def __init__(
        self,
        location=None,
        transport=get_json,
        today=None,
        climate=None,
        historical_only=False,
        forecast_only=False,
    ):
        from travel_agent.understanding.climatology import ClimateDatabase

        self.location = location
        self.transport = transport
        self.today = today
        self.climate = climate or ClimateDatabase()
        self.historical_only = historical_only
        self.forecast_only = forecast_only

    def fetch(self, request):
        from travel_agent.understanding.weather_locations import WeatherLocationResolver

        if not request.start_date or not request.end_date or not request.destination:
            return WeatherContext(
                reason="Destination and exact dates are required",
                reason_code="missing_weather_input",
                next_steps=["Provide a city and exact trip dates."],
            )
        try:
            cached = self.climate.records()
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            cached = []  # Bad historical storage must not prevent city clarification or forecasts.
        location = WeatherLocationResolver(self.transport, cached).resolve(
            request.destination, self.location
        )
        if isinstance(location, WeatherContext):
            return location
        try:
            today = self.today or datetime.now(ZoneInfo(location["timezone"])).date()
        except (ValueError, KeyError, TypeError):
            return WeatherContext(
                status="needs_location",
                reason="Invalid destination timezone",
                reason_code="incomplete_coordinates",
                next_steps=["Confirm a valid city timezone."],
            )
        outside = request.start_date < today or request.end_date > today + timedelta(days=15)
        if self.historical_only or outside:
            if self.forecast_only:
                return WeatherContext(
                    status="out_of_range",
                    reason_code="forecast_outside_window",
                    reason="Forecast-only mode cannot cover these dates",
                    next_steps=[
                        "Use automatic or historical mode for a labelled planning reference."
                    ],
                )
            boundary = (
                "explicit_historical_mode"
                if self.historical_only
                else (
                    "crosses_forecast_window"
                    if request.start_date <= today + timedelta(days=15)
                    and request.end_date >= today
                    else "outside_forecast_window"
                )
            )
            result = self.climate.fetch(request, location, routing_reason=boundary)
        else:
            result = OpenMeteoWeather(location, self.transport, today).fetch(request)
            result.routing_reason = "within_forecast_window"
            if result.status != "ok":
                result.reason_code = result.reason_code or "forecast_unavailable"
                result.next_steps = [
                    "Retry the forecast later; no historical result was substituted.",
                    "Explicitly select climatology if a historical reference is acceptable.",
                ]
        result.limitations.append(
            "City-level outdoor reference only; not indoor, attraction, mountain "
            "or beach microclimates."
        )
        return result


class HistoricalWeather:
    def __init__(self, location=None):
        self.provider = PlanningWeather(location=location, historical_only=True)

    def fetch(self, request):
        return self.provider.fetch(request)
