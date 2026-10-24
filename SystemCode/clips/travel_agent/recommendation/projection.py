"""Minimal request projection: Student 1 `TripShoppingRequest` -> Student 5 `TripShoppingRequest`.

Only the eight fields both modules define are projected, read from the request after
validation by Student 1's shared schema (so its defaults apply, e.g. empty `style` or
`excluded_items` lists). Student 1 extension fields are not forwarded, because Student 5 does
not consume them. No value is invented:

- A missing required field raises `RequestProjectionError` naming the fields.
- `user_size` is not set. Student 1 provides no size, and the source of `user_size` is still to
  be agreed by the team. With `user_size=None`, Student 5 does not enable the size constraint.
- `budget` stays in major currency units, as in both schemas.

The result is validated by Student 5's own `TripShoppingRequest`, which still forbids extra
fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from travel_agent.recommendation.student5.models import TripShoppingRequest
from travel_agent.schemas import TripShoppingRequest as Student1TripShoppingRequest

SHARED_REQUEST_FIELDS = (
    "destination",
    "start_date",
    "end_date",
    "activities",
    "budget",
    "currency",
    "style",
    "excluded_items",
)
# Student 5 requires these; Student 1 allows them to be empty while it is still collecting.
REQUIRED_REQUEST_FIELDS = ("destination", "start_date", "end_date", "budget", "currency")
# Student 1 extension fields that are deliberately not forwarded to Student 5.
STUDENT1_EXTENSION_FIELDS = (
    "owned_item_ids",
    "places",
    "dress_codes",
    "enter_interior",
    "gender",
    "cold_sensitive",
    "price_preference",
)


class RequestProjectionError(ValueError):
    """Student 1's request lacks fields that Student 5 requires."""

    def __init__(self, missing_fields: list[str]) -> None:
        self.missing_fields = missing_fields
        super().__init__(
            f"Student 1 request is missing fields Student 5 requires: {missing_fields}"
        )


def project_trip_request(trip_request: Mapping[str, Any]) -> TripShoppingRequest:
    """Project Student 1's request (e.g. `ModuleResult.data["trip_request"]`) onto the eight
    shared fields and validate it as a Student 5 `TripShoppingRequest`."""
    source = Student1TripShoppingRequest.model_validate(trip_request)
    missing = [name for name in REQUIRED_REQUEST_FIELDS if getattr(source, name) in (None, "")]
    if missing:
        raise RequestProjectionError(missing)
    return TripShoppingRequest.model_validate(
        {name: getattr(source, name) for name in SHARED_REQUEST_FIELDS}
    )
