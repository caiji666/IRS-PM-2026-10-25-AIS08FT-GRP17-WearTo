"""Lightweight checks for the public Student 4 service boundary."""

from travel_agent.outfits.service import run


def test_empty_payload_is_validated_by_dynamic_service_not_reported_as_scaffold():
    result = run({})

    assert result.status == "needs_input"
    assert "exactly one" in (result.reason or "").lower()
