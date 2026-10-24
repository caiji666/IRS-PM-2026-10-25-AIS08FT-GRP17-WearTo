"""Stage 13 — merge a TripShoppingRequestPatch into the current TripShoppingRequest.

Proposal (Appendix A): a MODIFY_REQUIREMENTS patch has `set`, `add` and `remove`; fields absent
from the patch keep their values, and a field is removed only when the user explicitly asks.

Merge semantics (the exact `add` / `remove` rules are PROVISIONAL):

- `set`: replaces the field value (a list field is replaced as a whole).
- `add`: list fields only (`activities`, `style`, `excluded_items`). Values are appended in
  patch order after the existing values; a value already present (compared after stripping
  surrounding whitespace, case-sensitive) or repeated within the patch is not added again.
- `remove`:
    * list field + [] → clear the field (team contract, docs/contracts.md: "Empty remove arrays
      explicitly clear that list");
    * list field + non-empty list of values → remove those values (whitespace-stripped,
      case-sensitive, every occurrence); values not present are ignored;
    * list field + null → clear the field too (Student 5's earlier form, kept for
      backward compatibility);
    * optional field (`user_size`) + null → reset it to null;
    * anything else (a required field, or a non-null value for a non-list field, or a non-list
      value for a list field) → PatchApplicationError. Nothing is guessed.

The merged request is re-validated as a TripShoppingRequest, so invalid dates, budget or
currency fail. `changed_fields` lists only fields whose value really changed, in
TripShoppingRequest field order; it is the input for the later selective rerun planner. This
module never decides or runs pipeline stages.

RequirementUpdateResult, PatchApplicationError and the function name are PROVISIONAL.
"""

from __future__ import annotations

from typing import Any

from pydantic import model_validator

from travel_agent.recommendation.student5.models import (
    TRIP_REQUEST_LIST_FIELDS,
    ContractModel,
    NonEmptyStr,
    RecommendationIntent,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)

REQUEST_FIELD_ORDER = tuple(TripShoppingRequest.model_fields)
OPTIONAL_SCALAR_FIELDS = frozenset({"user_size"})


class PatchApplicationError(ValueError):
    """PROVISIONAL. The patch cannot be applied without guessing its meaning."""


def changed_request_fields(
    previous: TripShoppingRequest, updated: TripShoppingRequest
) -> list[str]:
    """Fields whose value differs between two requests, in TripShoppingRequest field order."""
    return [f for f in REQUEST_FIELD_ORDER if getattr(previous, f) != getattr(updated, f)]


class RequirementUpdateResult(ContractModel):
    """PROVISIONAL. Before/after requests of one MODIFY_REQUIREMENTS patch."""

    previous_request: TripShoppingRequest
    updated_request: TripShoppingRequest
    changed_fields: list[NonEmptyStr]
    changed: bool

    @model_validator(mode="after")
    def _consistent(self) -> RequirementUpdateResult:
        expected = changed_request_fields(self.previous_request, self.updated_request)
        if self.changed_fields != expected:
            raise ValueError(f"changed_fields must be {expected}, got {self.changed_fields}")
        if self.changed != bool(expected):
            raise ValueError("changed must be true exactly when changed_fields is non-empty")
        return self


def _key(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _add_values(current: list[Any], values: list[Any]) -> list[Any]:
    merged = list(current)
    seen = {_key(v) for v in merged}
    for value in values:
        if _key(value) not in seen:
            merged.append(value)
            seen.add(_key(value))
    return merged


def _remove(field: str, current: Any, value: Any) -> Any:
    if field in TRIP_REQUEST_LIST_FIELDS:
        # Team contract (docs/contracts.md): an empty remove array clears the list.
        # null is still accepted as Student 5's earlier, equivalent form.
        if value is None or value == []:
            return []
        if isinstance(value, list):
            to_remove = {_key(v) for v in value}
            return [v for v in current if _key(v) not in to_remove]
        raise PatchApplicationError(
            f"remove.{field} must be a list of values to remove, or [] to clear the field"
        )
    if field in OPTIONAL_SCALAR_FIELDS:
        if value is None:
            return None
        raise PatchApplicationError(f"remove.{field} only accepts null (reset the field)")
    raise PatchApplicationError(f"remove.{field}: required field cannot be removed; use set")


def apply_requirement_patch(
    request: TripShoppingRequest, patch: TripShoppingRequestPatch
) -> RequirementUpdateResult:
    """Merge a MODIFY_REQUIREMENTS patch; inputs are not modified."""
    if not isinstance(patch, TripShoppingRequestPatch):
        raise TypeError(f"expected a TripShoppingRequestPatch, got {type(patch).__name__}")
    if patch.intent is not RecommendationIntent.MODIFY_REQUIREMENTS:
        raise PatchApplicationError(
            f"only MODIFY_REQUIREMENTS patches can be applied, got {patch.intent}"
        )

    data = request.model_dump()
    for field, value in patch.set.items():
        data[field] = value
    for field, values in patch.add.items():
        data[field] = _add_values(data[field], values)
    for field, value in patch.remove.items():
        data[field] = _remove(field, data[field], value)

    previous = TripShoppingRequest.model_validate(request.model_dump())
    updated = TripShoppingRequest.model_validate(data)
    changed = changed_request_fields(previous, updated)
    return RequirementUpdateResult(
        previous_request=previous,
        updated_request=updated,
        changed_fields=changed,
        changed=bool(changed),
    )
