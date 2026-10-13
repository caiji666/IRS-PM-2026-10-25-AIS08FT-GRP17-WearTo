"""Minimal request projection Student 1 -> Student 5 (`travel_agent.recommendation.projection`)."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from travel_agent.recommendation.projection import (
    REQUIRED_REQUEST_FIELDS,
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


def test_shared_fields_exist_in_both_schemas_and_extensions_only_in_student1():
    student1, student5 = Student1TripShoppingRequest.model_fields, TripShoppingRequest.model_fields
    assert set(SHARED_REQUEST_FIELDS) <= set(student1) & set(student5)
    assert set(STUDENT1_EXTENSION_FIELDS) <= set(student1)
    assert not set(STUDENT1_EXTENSION_FIELDS) & set(student5)
    assert set(student5) - set(SHARED_REQUEST_FIELDS) == {"user_size"}


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


@pytest.mark.parametrize("field", REQUIRED_REQUEST_FIELDS)
def test_missing_required_field_is_named_and_never_invented(student1_result, field):
    trip = dict(student1_result.data["trip_request"])
    trip[field] = None
    with pytest.raises(RequestProjectionError) as exc:
        project_trip_request(trip)
    assert exc.value.missing_fields == [field]
    assert field in str(exc.value)


def test_all_missing_required_fields_are_reported_together():
    with pytest.raises(RequestProjectionError) as exc:
        project_trip_request({"activities": ["sightseeing"]})
    assert exc.value.missing_fields == list(REQUIRED_REQUEST_FIELDS)


@pytest.mark.parametrize("omitted", ["activities", "style", "excluded_items"])
def test_omitted_list_fields_use_the_student1_schema_default(student1_result, omitted):
    trip = dict(student1_result.data["trip_request"])
    del trip[omitted]
    assert getattr(project_trip_request(trip), omitted) == []


def test_invalid_student1_request_is_rejected_by_its_own_schema(student1_result):
    trip = dict(student1_result.data["trip_request"])
    trip["currency"] = "XYZ"
    with pytest.raises(ValueError):
        project_trip_request(trip)


def test_projection_does_not_mutate_the_input(student1_result):
    trip = dict(student1_result.data["trip_request"])
    before = json.dumps(trip, sort_keys=True, default=str)
    project_trip_request(trip)
    assert json.dumps(trip, sort_keys=True, default=str) == before
