"""Minimal request projection Student 1 -> Student 5 (`travel_agent.recommendation.projection`)."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from travel_agent.recommendation.projection import (
    SHARED_REQUEST_FIELDS,
    STUDENT1_EXTENSION_FIELDS,
    RequestProjectionError,
    project_trip_request,
)
from travel_agent.recommendation.student5.models import TripShoppingRequest
from travel_agent.schemas import SUPPORTED_CURRENCIES
from travel_agent.schemas import TripShoppingRequest as Student1TripShoppingRequest
from travel_agent.understanding.service import run as student1_run

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def student1_result():
    payload = json.loads(
        (ROOT / "examples/understanding/seoul_winter.json").read_text(encoding="utf-8")
    )
    result = student1_run(payload)
    assert result.status == "ok", result.reason
    return result


def test_real_student1_output_projects_onto_the_shared_fields(student1_result):
    trip = student1_result.data["trip_request"]
    projected = project_trip_request(trip)
    assert isinstance(projected, TripShoppingRequest)
    source = Student1TripShoppingRequest.model_validate(trip)
    for name in SHARED_REQUEST_FIELDS:
        assert getattr(projected, name) == getattr(source, name), name
    assert projected.user_size is None


def test_budget_stays_in_major_units_and_matches_student1_budget_minor(student1_result):
    projected = project_trip_request(student1_result.data["trip_request"])
    digits = SUPPORTED_CURRENCIES[projected.currency]
    assert projected.budget == Decimal("500")
    assert (
        projected.budget.scaleb(digits) == student1_result.data["hard_constraints"]["budget_minor"]
    )


def test_student1_extension_fields_are_not_forwarded(student1_result):
    trip = dict(student1_result.data["trip_request"])
    trip.update(
        owned_item_ids=["coat-1"],
        places=["Gyeongbokgung Palace"],
        dress_codes={"dinner": "smart casual"},
        enter_interior=True,
        gender="female",
        cold_sensitive=True,
        price_preference="lower",
    )
    projected = project_trip_request(trip)
    dumped = projected.model_dump()
    assert not set(STUDENT1_EXTENSION_FIELDS) & set(dumped)
    assert set(dumped) == set(SHARED_REQUEST_FIELDS) | {"user_size"}


@pytest.mark.parametrize("field", ["destination", "budget"])
def test_missing_required_field_is_named_and_never_invented(student1_result, field):
    trip = dict(student1_result.data["trip_request"])
    trip[field] = None
    with pytest.raises(RequestProjectionError) as exc:
        project_trip_request(trip)
    assert exc.value.missing_fields == [field]
    assert field in str(exc.value)


@pytest.mark.parametrize("omitted", ["activities"])
def test_omitted_list_fields_use_the_student1_schema_default(student1_result, omitted):
    trip = dict(student1_result.data["trip_request"])
    del trip[omitted]
    assert getattr(project_trip_request(trip), omitted) == []
