import json
from pathlib import Path

import pytest

from travel_agent.schemas import TripShoppingRequest, WeatherContext
from travel_agent.understanding.season import derive_season
from travel_agent.understanding.service import run


@pytest.mark.parametrize(
    "city,expected", [("Seoul", "winter"), ("首尔", "winter"), ("Sydney", "summer")]
)
def test_hemisphere_and_supported_alias(city, expected):
    request = TripShoppingRequest(destination=city, start_date="2026-12-20", end_date="2026-12-27")
    result = derive_season(request)
    assert result.season == expected
    assert result.status == "derived"
    assert result.sources and result.review_status == "pending"


def test_cross_season_and_cross_year():
    request = TripShoppingRequest(
        destination="Seoul", start_date="2026-11-30", end_date="2026-12-02"
    )
    result = derive_season(request)
    assert result.season is None and result.seasons == ["autumn", "winter"]
    assert result.status == "multiple"
    request = TripShoppingRequest(
        destination="Seoul", start_date="2026-12-30", end_date="2027-01-03"
    )
    assert derive_season(request).season == "winter"


def test_unknown_dates_location_and_tropical_scope():
    assert derive_season(TripShoppingRequest(destination="Seoul")).season is None
    request = TripShoppingRequest(
        destination="Unknown city", start_date="2026-12-20", end_date="2026-12-27"
    )
    assert derive_season(request).status == "unknown"
    request.destination = "Singapore"
    result = derive_season(request)
    assert result.season is None and result.status == "unsupported"


def test_calendar_season_is_not_temperature_class():
    request = TripShoppingRequest(
        destination="Seoul", start_date="2026-12-20", end_date="2026-12-27"
    )
    weather = WeatherContext(weather_class="HOT")
    assert derive_season(request, weather).season == "winter"
    assert derive_season(request, WeatherContext(status="needs_location")).season is None


def test_coordinate_scope_and_weather_provider_precedence():
    request = TripShoppingRequest(
        destination="New city", start_date="2026-12-20", end_date="2026-12-27"
    )
    assert (
        derive_season(
            request, confirmed_location={"destination": "Other city", "latitude": -35}
        ).season
        is None
    )
    assert (
        derive_season(
            request, confirmed_location={"destination": "New city", "latitude": -35}
        ).season
        == "summer"
    )


def test_output_season_and_embedding_text_without_changing_user_request():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "examples/understanding/seoul_winter.json").read_text())
    result = run(payload)
    assert result.data["season"] == "winter"
    assert result.data["expanded_fashion_context"].startswith("Seoul winter cold ")
    assert "season" not in result.data["trip_request"]
    assert (
        result.data["enriched_trip_context"]["weather_context"]["data_type"] == "synthetic_fixture"
    )


@pytest.mark.parametrize("latitude", ["35", True, float("nan"), 91])
def test_invalid_coordinates_do_not_crash_or_invent_season(latitude):
    request = TripShoppingRequest(
        destination="New city", start_date="2026-12-20", end_date="2026-12-27"
    )
    result = derive_season(
        request, confirmed_location={"destination": "New city", "latitude": latitude}
    )
    assert result.season is None and result.status == "unknown"
