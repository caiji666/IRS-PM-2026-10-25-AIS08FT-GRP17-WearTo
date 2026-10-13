"""Stage 13 — Selective Re-execution Planner.

Proposal: after a requirement change the updated request is passed through *the relevant*
recommendation stages, and unchanged requirements are preserved. This module only decides
which stages those are; it never calls, simulates or schedules any pipeline stage, and never
touches the request or the conversation state.

Input is RequirementUpdateResult.changed_fields (TripShoppingRequest field names, computed by
patch_processor). Output is a RerunPlan: the union of the per-field stage lists, sorted and
de-duplicated, starting from the lowest affected stage, with a reason generated from the
mapping.

The mapping is a conservative dependency expansion: once a stage's output may change, every
downstream stage that consumes it is re-run too. In particular:

1. Stage 4 (search queries / context embedding) changes → Stages 5 and 6 refresh the social
   references, because Stage 8 scores products against them.
2. Stage 8 (candidate product pool / product data) changes → Stage 9 regenerates the
   candidate outfits from the new pools.
3. Stage 9 or 10 may change the feasible outfit set → Stage 11 computes CompatibilityScore for
   the new feasible outfits.
4. Stage 12 never receives an outfit with compatibility_score=None (ranking_engine rejects it),
   which rule 3 guarantees.
5. If the team later confirms that cached results can be reused safely, individual mappings can
   be shortened.

The field → stage mapping, RerunPlan, NoChangesError and the function names are all
PROVISIONAL (the proposal only says "re-execute the relevant recommendation stages").
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, StrictInt, field_validator, model_validator

from travel_agent.recommendation.student5.models import ContractModel, NonEmptyStr
from travel_agent.recommendation.student5.patch_processor import (
    REQUEST_FIELD_ORDER,
    RequirementUpdateResult,
)

_FROM_STAGE_2 = tuple(range(2, 13))
_FROM_STAGE_8 = (8, 9, 10, 11, 12)

# PROVISIONAL field → affected stages mapping (conservative dependency expansion, see above).
FIELD_STAGE_MAP: dict[str, tuple[int, ...]] = {
    "destination": _FROM_STAGE_2,
    "start_date": _FROM_STAGE_2,
    "end_date": _FROM_STAGE_2,
    "activities": (3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "budget": (10, 11, 12),
    "currency": _FROM_STAGE_8,
    "style": (4, 5, 6, 7, 8, 9, 10, 11, 12),
    "excluded_items": _FROM_STAGE_8,
    "user_size": _FROM_STAGE_8,
}

# Stage names follow the proposal's 14-stage pipeline (Section 7.1); proposal headings for
# 5, 6 and 11 are "Social Media / Web Style Retrieval", "Social Post Reranking" and
# "Outfit Compatibility Model" (11B Online Outfit Scoring).
STAGE_NAMES: dict[int, str] = {
    2: "weather context enrichment",
    3: "hybrid context reasoning",
    4: "query composition and context embedding",
    5: "social reference retrieval",
    6: "social reference reranking",
    7: "e-commerce product retrieval",
    8: "product filtering and reranking",
    9: "candidate outfit generation",
    10: "hard-constraint filtering",
    11: "outfit compatibility scoring",
    12: "final ranking",
}


class NoChangesError(ValueError):
    """PROVISIONAL. Nothing changed, so there is no rerun plan to build."""


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _canonical_fields(fields: Iterable[str]) -> list[str]:
    present = set(fields)
    return [f for f in REQUEST_FIELD_ORDER if f in present]


def affected_stages_for(fields: Iterable[str]) -> list[int]:
    """Sorted union of the mapped stages of known fields."""
    return sorted({stage for field in fields for stage in FIELD_STAGE_MAP[field]})


def build_reason(changed_fields: list[str]) -> str:
    """Deterministic reason: one clause per field, in canonical field order."""
    clauses = [
        f"{field} affects "
        + _join([f"{STAGE_NAMES[s]} (Stage {s})" for s in FIELD_STAGE_MAP[field]])
        for field in changed_fields
    ]
    return "; ".join(clauses)


class RerunPlan(ContractModel):
    """PROVISIONAL. Which pipeline stages must re-run after a requirement change."""

    changed_fields: list[NonEmptyStr] = Field(min_length=1)
    rerun_from_stage: StrictInt
    affected_stages: list[StrictInt] = Field(min_length=1)
    reason: NonEmptyStr

    @field_validator("changed_fields")
    @classmethod
    def _known_unique_canonical(cls, fields: list[str]) -> list[str]:
        unknown = [f for f in fields if f not in FIELD_STAGE_MAP]
        if unknown:
            raise ValueError(f"unknown TripShoppingRequest field(s): {unknown}")
        if fields != _canonical_fields(fields) or len(set(fields)) != len(fields):
            raise ValueError("changed_fields must be unique and in TripShoppingRequest order")
        return fields

    @field_validator("affected_stages")
    @classmethod
    def _sorted_unique(cls, stages: list[int]) -> list[int]:
        if stages != sorted(set(stages)):
            raise ValueError("affected_stages must be strictly ascending without duplicates")
        return stages

    @model_validator(mode="after")
    def _consistent_with_mapping(self) -> RerunPlan:
        expected = affected_stages_for(self.changed_fields)
        if self.affected_stages != expected:
            raise ValueError(f"affected_stages must be {expected} for {self.changed_fields}")
        if self.rerun_from_stage != self.affected_stages[0]:
            raise ValueError("rerun_from_stage must equal the smallest affected stage")
        if self.reason != build_reason(self.changed_fields):
            raise ValueError("reason must be generated from the field mapping")
        return self


def plan_selective_rerun(changed_fields: Iterable[str]) -> RerunPlan:
    """Build the RerunPlan for the given changed TripShoppingRequest fields.

    Duplicates are removed and fields are put in TripShoppingRequest order; the input is not
    modified. Raises NoChangesError for no fields and ValueError for unknown fields.
    """
    if isinstance(changed_fields, str):
        raise TypeError("changed_fields must be a collection of field names, not a string")
    fields = list(changed_fields)
    unknown = [f for f in fields if f not in FIELD_STAGE_MAP]
    if unknown:
        raise ValueError(f"unknown TripShoppingRequest field(s): {unknown}")
    canonical = _canonical_fields(fields)
    if not canonical:
        raise NoChangesError("no changed fields, so nothing needs to be re-run")
    stages = affected_stages_for(canonical)
    return RerunPlan(
        changed_fields=canonical,
        rerun_from_stage=stages[0],
        affected_stages=stages,
        reason=build_reason(canonical),
    )


def plan_from_update_result(update_result: RequirementUpdateResult) -> RerunPlan:
    """Build the RerunPlan from a MODIFY_REQUIREMENTS result; NoChangesError if unchanged."""
    if not isinstance(update_result, RequirementUpdateResult):
        raise TypeError(f"expected a RequirementUpdateResult, got {type(update_result).__name__}")
    if not update_result.changed:
        raise NoChangesError(
            "the requirement update changed nothing, so nothing needs to be re-run"
        )
    return plan_selective_rerun(update_result.changed_fields)
