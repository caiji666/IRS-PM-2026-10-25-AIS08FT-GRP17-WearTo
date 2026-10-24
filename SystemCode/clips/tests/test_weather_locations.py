"""Country/city clarification, bounded acquisition and conservative geographic scope."""

import copy
import json
import sys
from datetime import date
from urllib.parse import parse_qs, urlparse

import pytest

from travel_agent.schemas import TripShoppingRequest
from travel_agent.understanding.climatology import ClimateDatabase, cache_records
from travel_agent.understanding.weather import PlanningWeather
from travel_agent.understanding.weather_locations import WeatherLocationResolver, load_registry


def no_network(url):
    raise AssertionError("No network is needed for this clarification/cached case")


def request(destination):
    return TripShoppingRequest(
        destination=destination, start_date="2027-02-01", end_date="2027-02-04"
    )


@pytest.fixture
def climate(tmp_path):
    return ClimateDatabase(db=tmp_path / "climate.sqlite3")


def provider(climate, location=None, transport=no_network):
    return PlanningWeather(
        climate=climate, location=location, transport=transport, today=date(2026, 10, 21)
    )


@pytest.mark.parametrize(
    "country", ["日本", "Japan", "JP", "中国", "Malaysia", "英国", "United States"]
)
def test_countries_need_city_selection_without_weather_guess(climate, country):
    weather = provider(climate).fetch(request(country))
    assert weather.status == "needs_location" and weather.reason_code == "country_requires_city"
    assert weather.location_candidates
    assert weather.temp_min_celsius is None and weather.climate_reference is None


def test_all_fifteen_city_references_work_offline(climate):
    for city in load_registry()["records"]:
        destination = city["name"]
        weather = provider(climate).fetch(request(destination))
        assert weather.status == "ok", (destination, weather.reason)
        assert weather.data_type == "historical_climatology"
        assert weather.destination == destination and weather.location_label == destination
        assert weather.precipitation_probability is None
        assert weather.climate_reference.location_basis == "registered_city_default"


def test_explicit_country_city_selection_keeps_original_destination(climate):
    tokyo = next(c for c in load_registry()["records"] if c["name"] == "Tokyo")
    weather = provider(climate, dict(tokyo, destination="日本")).fetch(request("日本"))
    assert weather.status == "ok" and weather.destination == "日本"
    assert weather.location_label == "Tokyo"
    assert weather.climate_reference.location_basis == "caller_confirmed_coordinates"
    paris = next(c for c in load_registry()["records"] if c["name"] == "Paris")
    wrong = provider(climate, dict(paris, destination="日本")).fetch(request("日本"))
    assert wrong.status == "needs_location" and wrong.reason_code == "country_mismatch"


@pytest.mark.parametrize("name", ["Mount Fuji", "富士山", "Bali", "巴厘岛", "indoor museum"])
def test_regions_and_venues_are_not_silently_city_weather(climate, name):
    weather = provider(climate).fetch(request(name))
    assert weather.status == "needs_location" and weather.reason_code == "non_city_scope"
    assert weather.data_type is None


def test_nearby_point_and_elevation_mismatch_do_not_reuse_city(climate):
    tokyo = next(c for c in load_registry()["records"] if c["name"] == "Tokyo")
    moved = dict(tokyo, destination="Tokyo", latitude=tokyo["latitude"] + 0.001)
    weather = provider(climate, moved).fetch(request("Tokyo"))
    assert weather.status == "out_of_range" and weather.reason_code == "history_not_cached"
    assert weather.acquisition["max_locations"] == 1
    high = dict(tokyo, destination="Tokyo", elevation=tokyo["elevation"] + 1000)
    assert provider(climate, high).fetch(request("Tokyo")).reason_code == "elevation_mismatch"
    indoor = dict(tokyo, destination="Tokyo", scope="indoor")
    assert provider(climate, indoor).fetch(request("Tokyo")).reason_code == "non_city_scope"


def test_unknown_city_geocodes_once_without_automatic_archive_download(climate):
    calls = []

    def geocode(url):
        calls.append(url)
        assert "geocoding-api" in url and "archive-api" not in url
        assert parse_qs(urlparse(url).query)["countryCode"] == ["FR"]
        return {
            "results": [
                {
                    "name": "Testville",
                    "country_code": "FR",
                    "feature_code": "PPL",
                    "latitude": 44.0,
                    "longitude": 4.0,
                    "timezone": "Europe/Paris",
                }
            ]
        }

    weather = provider(climate, transport=geocode).fetch(request("Testville, France"))
    assert len(calls) == 1
    assert weather.reason_code == "history_not_cached"
    assert weather.acquisition["requires_explicit_command"] is True
    assert weather.acquisition["location"]["destination"] == "Testville, France"
    assert "cache-location" in weather.acquisition["command"]


def test_unknown_same_name_and_geocoding_failure_have_clear_next_steps(climate):
    def transport(url):
        return {
            "results": [
                {"name": "Springfield", "feature_code": "PPL"},
                {"name": "Springfield", "feature_code": "PPL"},
            ]
        }

    weather = provider(climate, transport=transport).fetch(request("Springfield"))
    assert weather.status == "needs_location" and weather.reason_code == "ambiguous_location"
    assert len(weather.location_candidates) == 2
    assert all(c["source"].startswith("https://geocoding-api") for c in weather.location_candidates)

    def offline(url):
        raise OSError("offline")

    weather = provider(climate, transport=offline).fetch(request("Testville, France"))
    assert weather.status == "unknown" and weather.reason_code == "geocoding_unavailable"
    assert weather.next_steps and weather.data_type is None


def test_non_registry_country_result_never_becomes_a_city():
    resolver = WeatherLocationResolver(
        lambda url: {"results": [{"name": "Italy", "feature_code": "PCLI"}]}
    )
    weather = resolver.resolve("Italy")
    assert weather.status == "needs_location" and weather.reason_code == "country_requires_city"


def test_country_qualified_alias_and_stale_city_are_handled(climate):
    assert provider(climate).fetch(request("东京，日本")).status == "ok"
    paris = next(c for c in load_registry()["records"] if c["name"] == "Paris")
    wrong = provider(climate, dict(paris, destination="Tokyo")).fetch(request("Tokyo"))
    assert wrong.reason_code == "city_mismatch"


def test_service_country_question_preserves_user_facts(monkeypatch, tmp_path):
    from pathlib import Path

    from travel_agent.understanding.service import run

    monkeypatch.setenv("WEATHER_CLIMATE_DB", str(tmp_path / "service.sqlite3"))
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "examples/understanding/seoul_winter.json").read_text())
    payload.pop("weather_fixture")
    payload["request"]["destination"] = "日本"
    payload["weather_provider"] = "open_meteo"
    result = run(payload)
    assert result.status == "needs_input" and not result.data["ready_for_downstream"]
    assert result.data["trip_request"]["destination"] == "日本"
    assert result.data["questions"][0]["reason_code"] == "country_requires_city"


def test_cache_upserts_preserve_other_cities_and_same_name_foreign_city(climate):
    original = climate.records()
    # Synthetic point record tests key separation; it is never part of the real shared snapshot.
    extra = copy.deepcopy(next(r for r in original if r["name"] == "Paris"))
    extra.update(country_code="US", latitude=33.66, longitude=-95.55, timezone="America/Chicago")
    cache_records([extra], climate.db)
    records = climate.records()
    assert len(records) == 16
    assert len([r for r in records if r["name"] == "Paris"]) == 2
    assert provider(climate).fetch(request("Paris")).reason_code == "ambiguous_location"
    assert provider(climate).fetch(request("Paris, France")).status == "ok"
    cache_records(original, climate.db)
    assert len(climate.records()) == 16  # Merge preserves the added point.


def test_failed_explicit_acquisition_leaves_database_untouched(monkeypatch, tmp_path, capsys):
    from travel_agent.understanding import climatology

    city = next(c for c in load_registry()["records"] if c["name"] == "Tokyo")
    path = tmp_path / "confirmed.json"
    path.write_text(json.dumps(city | {"destination": "Tokyo"}))
    db = tmp_path / "not-created.sqlite3"

    def fail(*args, **kwargs):
        raise OSError("historical API unavailable")

    monkeypatch.setattr(climatology, "acquire_city", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        ["climatology", "cache-location", "--location-file", str(path), "--db", str(db)],
    )
    with pytest.raises(SystemExit) as error:
        climatology.main()
    assert error.value.code == 1 and not db.exists()
    assert "historical_acquisition_failed" in capsys.readouterr().err
