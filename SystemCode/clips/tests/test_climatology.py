"""Routing boundaries, real snapshot integrity and truthful historical rule semantics."""

import calendar
import copy
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

from travel_agent.schemas import TripShoppingRequest, WeatherContext
from travel_agent.understanding.climatology import (
    SEED,
    ClimateDatabase,
    aggregate,
    import_seed,
    validate_record,
)
from travel_agent.understanding.rules import reason_weather
from travel_agent.understanding.service import run
from travel_agent.understanding.weather import OpenMeteoWeather, PlanningWeather


@pytest.fixture
def climate(tmp_path):
    return ClimateDatabase(db=tmp_path / "climate.sqlite3")


def request(start="2026-12-20", end="2026-12-27", destination="Seoul"):
    return TripShoppingRequest(destination=destination, start_date=start, end_date=end)


def no_network(url):
    raise AssertionError("Historical lookup must not call any API")


def forecast_transport(url):
    """Explicit synthetic transport, used only to verify routing/coverage validation."""
    query = parse_qs(urlparse(url).query)
    # Official API rejects forecast_days together with explicit start/end dates.
    assert "forecast_days" not in query
    start, end = (date.fromisoformat(query[k][0]) for k in ("start_date", "end_date"))
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return {
        "daily_units": {
            "temperature_2m_min": "°C",
            "temperature_2m_max": "°C",
            "precipitation_probability_max": "%",
            "snowfall_sum": "cm",
        },
        "daily": {
            "time": [d.isoformat() for d in dates],
            "temperature_2m_min": [12] * len(dates),
            "temperature_2m_max": [20] * len(dates),
            "precipitation_probability_max": [40] * len(dates),
            "snowfall_sum": [0] * len(dates),
        },
    }


def test_seed_has_complete_real_scoped_references(climate):
    assert not climate.db.exists()
    records = climate.records()
    assert climate.db.exists()
    assert {r["name"] for r in records} == {
        "Seoul",
        "Tokyo",
        "Singapore",
        "Osaka",
        "Beijing",
        "Shanghai",
        "Kuala Lumpur",
        "Bangkok",
        "Hanoi",
        "Ho Chi Minh City",
        "Denpasar",
        "London",
        "Paris",
        "Sydney",
        "New York",
    }
    assert len({r["country_code"] for r in records}) == 12
    for record in records:
        validate_record(record)
        assert record["baseline_start_year"] == 2015
        assert record["baseline_end_year"] == 2024
        assert len(record["raw_response_sha256"]) == 64
        assert record["source_urls"][0].startswith("https://archive-api.open-meteo.com/")
        assert sum(m["sample_days"] for m in record["monthly"]) == 3653


@pytest.mark.parametrize("start,end", [("2026-10-21", "2026-10-21"), ("2026-11-05", "2026-11-05")])
def test_today_and_fifteenth_day_are_forecasts(climate, start, end):
    weather = PlanningWeather(
        climate=climate, today=date(2026, 10, 21), transport=forecast_transport
    ).fetch(request(start, end))
    assert weather.status == "ok" and weather.data_type == "forecast"
    assert weather.climate_reference is None
    assert weather.precipitation_probability == 0.4
    assert weather.location_basis == "registered_city_default"


@pytest.mark.parametrize(
    "start,end,reason",
    [
        ("2026-11-06", "2026-11-10", "outside_forecast_window"),
        ("2026-11-05", "2026-11-06", "crosses_forecast_window"),
        ("2026-10-20", "2026-10-22", "crosses_forecast_window"),
        ("2025-01-01", "2025-01-02", "outside_forecast_window"),
    ],
)
def test_entire_outside_or_crossing_trip_uses_only_history(climate, start, end, reason):
    weather = PlanningWeather(
        climate=climate, today=date(2026, 10, 21), transport=no_network
    ).fetch(request(start, end))
    assert weather.status == "ok" and weather.data_type == "historical_climatology"
    assert weather.routing_reason == reason
    assert weather.temp_min_celsius is None and weather.temp_max_celsius is None
    assert weather.precipitation_probability is None and weather.snow is None


def test_cross_year_and_leap_months_keep_monthly_references_separate(climate):
    weather = climate.fetch(request("2027-12-20", "2028-03-02"))
    monthly = weather.climate_reference.monthly
    assert [m.month for m in monthly] == [12, 1, 2, 3]
    assert monthly[2].sample_days == 283  # Three leap Februaries in 2015-2024.
    assert len(climate.fetch(request("2027-01-01", "2030-12-31")).climate_reference.monthly) == 12


def test_historical_temperatures_are_planning_suggestions_not_rain_or_snow_rules(climate):
    weather, clothes, trace = reason_weather(climate.fetch(request()))
    assert weather.weather_class == "COLD"
    assert "winter coat" in clothes.outerwear
    assert {t["rule_id"] for t in trace} == {"COLD"}
    assert trace[0]["field"] == "monthly_mean_daily_min_celsius"
    assert trace[0]["baseline_start_year"] == 2015
    assert weather.precipitation_probability is None and weather.snow is None


def test_unknown_location_and_changed_coordinates_are_not_substituted(climate):
    assert climate.fetch(request(destination="Atlantis")).status == "out_of_range"
    changed = {"destination": "Seoul", "latitude": 35.0, "longitude": 126.0}
    assert climate.fetch(request(), changed).status == "out_of_range"
    stale = changed | {"destination": "Tokyo"}
    assert climate.fetch(request(), stale).status == "needs_location"
    assert climate.fetch(request(destination="首尔")).status == "ok"


def test_api_failure_is_not_silently_replaced_by_history(climate):
    def fail(url):
        raise OSError("offline")

    weather = PlanningWeather(climate=climate, today=date(2026, 10, 21), transport=fail).fetch(
        request("2026-10-22", "2026-10-24")
    )
    assert weather.status == "unknown" and weather.climate_reference is None
    assert weather.routing_reason == "within_forecast_window"


def test_forecast_only_rejects_past_even_without_location_timezone():
    location = {"destination": "Seoul", "latitude": 37.566, "longitude": 126.9784}
    weather = OpenMeteoWeather(location=location, transport=no_network).fetch(
        request("2000-01-01", "2000-01-02")
    )
    assert weather.status == "out_of_range"


def test_routing_uses_destination_local_today(climate, monkeypatch):
    from travel_agent.understanding import weather as weather_module

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            # Seoul is already October 22; Singapore is still October 21.
            instant = datetime(2026, 10, 21, 15, 30, tzinfo=timezone.utc)
            return instant.astimezone(tz)

    monkeypatch.setattr(weather_module, "datetime", FrozenDateTime)
    weather = PlanningWeather(climate=climate, transport=forecast_transport).fetch(
        request("2026-11-06", "2026-11-06")
    )
    assert weather.status == "ok" and weather.data_type == "forecast"


def test_unknown_near_term_city_requires_geocoding_disambiguation(climate):
    weather = PlanningWeather(
        climate=climate,
        transport=lambda url: {"results": [{"name": "Springfield"}, {"name": "Other"}]},
    ).fetch(request(date.today().isoformat(), date.today().isoformat(), "Springfield"))
    assert weather.status == "needs_location"


def test_incomplete_actual_forecast_is_unknown_not_historical(climate):
    def incomplete(url):
        result = forecast_transport(url)
        result["daily"]["time"].pop()
        return result

    weather = PlanningWeather(
        climate=climate, today=date(2026, 10, 21), transport=incomplete
    ).fetch(request("2026-10-22", "2026-10-24"))
    assert weather.status == "unknown" and weather.data_type is None


def test_unavailable_database_is_explicit_and_does_not_prevent_live_forecast(tmp_path):
    db = tmp_path / "broken.sqlite3"
    db.write_text("not a database")
    climate = ClimateDatabase(db=db)
    assert climate.fetch(request()).status == "unknown"
    location = {"destination": "Seoul", "latitude": 37.566, "longitude": 126.9784}
    weather = PlanningWeather(
        climate=climate, location=location, today=date(2026, 10, 21), transport=forecast_transport
    ).fetch(request("2026-10-22", "2026-10-22"))
    assert weather.status == "ok" and weather.data_type == "forecast"


def test_invalid_import_preserves_existing_database(climate, tmp_path):
    records = climate.records()
    broken = copy.deepcopy(records)
    broken[0]["monthly"].pop()
    seed = tmp_path / "bad.json"
    seed.write_text(json.dumps({"schema_version": 1, "records": broken}))
    with pytest.raises(ValueError):
        import_seed(seed, climate.db)
    assert climate.records() == records
    with sqlite3.connect(climate.db) as connection:
        assert connection.execute("SELECT count(*) FROM cities").fetchone()[0] == 15


def test_historical_probability_and_snow_boolean_are_rejected(climate):
    data = climate.fetch(request()).model_dump()
    for wrong in ({"precipitation_probability": 0.7}, {"snow": False}):
        with pytest.raises(ValidationError):
            WeatherContext.model_validate(data | wrong)


def test_service_routes_far_dates_and_preserves_review_semantics(monkeypatch, tmp_path):
    monkeypatch.setenv("WEATHER_CLIMATE_DB", str(tmp_path / "new.sqlite3"))
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "examples/understanding/seoul_winter.json").read_text())
    payload.pop("weather_fixture")
    payload["weather_provider"] = "climatology"
    result = run(payload)
    weather = result.data["enriched_trip_context"]["weather_context"]
    assert result.status == "ok" and weather["data_type"] == "historical_climatology"
    assert "not a trip-date forecast" in result.data["expanded_fashion_context"]
    assert result.data["context_status"] == "partial"
    assert all(t["review_status"] == "pending" for t in result.data["weather_rule_trace"])


def test_missing_optional_precipitation_is_unknown_in_reference():
    # Explicit synthetic archive response tests aggregation, not real climate accuracy.
    city = json.loads(SEED.read_text())["records"][0]
    city["source"] = city["source_urls"][1]
    first, last = date(2015, 1, 1), date(2024, 12, 31)
    dates = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    count = len(dates)
    payload = {
        "timezone": "Asia/Seoul",
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "daily_units": {
            "temperature_2m_min": "°C",
            "temperature_2m_max": "°C",
            "precipitation_sum": "mm",
            "snowfall_sum": "cm",
        },
        "daily": {
            "time": [d.isoformat() for d in dates],
            "temperature_2m_min": [10] * count,
            "temperature_2m_max": [20] * count,
            "precipitation_sum": [None] * count,
            "snowfall_sum": [None] * count,
        },
    }
    record = aggregate(city, payload, "https://example.test/synthetic", 2015, 2024)
    assert record["monthly"][0]["wet_day_frequency"] is None
    assert record["monthly"][0]["snowfall_day_frequency"] is None
    assert record["monthly"][1]["sample_days"] == sum(
        calendar.monthrange(y, 2)[1] for y in range(2015, 2025)
    )
    payload["daily"]["temperature_2m_min"][0] = None
    with pytest.raises(ValueError):
        aggregate(city, payload, "https://example.test/synthetic", 2015, 2024)
