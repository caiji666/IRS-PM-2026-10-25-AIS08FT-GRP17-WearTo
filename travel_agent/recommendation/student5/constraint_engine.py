"""Stage 10 — Hard Constraint Filtering.

Implements the proposal's feasibility rule:

    Feasible(o) =
        [TotalPrice(o) <= Budget]
        AND [AllItemsAvailable(o)]
        AND [SizeConstraintSatisfied(o), if enabled]

Only purchase items (is_user_owned=False) count toward TotalPrice and are checked for
availability and size; the user's existing clothes are exempt. The size constraint is enabled
when the request has a user_size, and is applied only to products whose size_data_reliable
is True. Constraints are never relaxed: every violated constraint is recorded, and when no
outfit is feasible a structured conflict explanation is returned so the user can adjust
their requirements.

This stage runs before Stage 11, so each rejected outfit saves one CompatibilityScore call.
That count is a statistic only; downstream stages consume `feasible_outfits` alone.

The result structures (ConstraintType, RejectionReason, RejectedOutfit, ConstraintConflict,
ConstraintFilterResult, CurrencyMismatchError) are PROVISIONAL: the proposal only names the
output "Feasible Candidate Outfits".
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    ContractModel,
    FeasibleCandidateOutfit,
    Money,
    NonEmptyStr,
    Product,
    TripShoppingRequest,
)


class CurrencyMismatchError(ValueError):
    """PROVISIONAL. A purchase item's currency differs from the request currency.

    Currency normalisation is PROVISIONAL upstream work (proposal Section 5.3.4 says prices are
    normalised to the budget currency); this stage never converts currencies or compares
    amounts across currencies.
    """


class ConstraintType(StrEnum):
    """PROVISIONAL names for the three hard constraints of Stage 10."""

    BUDGET = "BUDGET"
    AVAILABILITY = "AVAILABILITY"
    SIZE = "SIZE"


class RejectionReason(ContractModel):
    """PROVISIONAL. One violated constraint of a rejected outfit."""

    constraint: ConstraintType
    message: NonEmptyStr
    product_ids: list[str] = Field(default_factory=list)


class RejectedOutfit(ContractModel):
    """PROVISIONAL. An outfit removed in Stage 10 with every reason it failed."""

    outfit_id: NonEmptyStr
    reasons: list[RejectionReason] = Field(min_length=1)
    total_purchase_cost: Money

    @field_validator("reasons")
    @classmethod
    def _one_reason_per_constraint(cls, reasons: list[RejectionReason]) -> list[RejectionReason]:
        counts = Counter(r.constraint for r in reasons)
        duplicated = sorted(c for c, n in counts.items() if n > 1)
        if duplicated:
            raise ValueError(f"duplicate rejection reason for constraint(s): {duplicated}")
        return reasons


class ConstraintConflict(ContractModel):
    """PROVISIONAL. Structured explanation returned when no candidate outfit is feasible.

    The constraints are not relaxed. `min_budget_to_make_feasible` is the lowest purchase
    total among outfits rejected *only* by the budget, i.e. information for the user to
    decide whether to raise the budget; it is None when raising the budget alone cannot help.
    """

    violated_constraints: list[ConstraintType] = Field(min_length=1)
    budget: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
    currency: NonEmptyStr
    min_budget_to_make_feasible: Money | None = None
    message: NonEmptyStr


class ConstraintFilterResult(ContractModel):
    """PROVISIONAL. Stage 10 output.

    `conflict_summary` counts, per constraint, how many outfits violated it.
    `compatibility_calls_saved` equals `rejected_count`, since Stage 11 is only run on feasible
    outfits; it is a statistic, not an input for later stages. `conflict` is present exactly
    when there were candidates but none is feasible.
    """

    feasible_outfits: list[FeasibleCandidateOutfit]
    rejected_outfits: list[RejectedOutfit]
    total_candidates: int = Field(ge=0)
    feasible_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    compatibility_calls_saved: int = Field(ge=0)
    conflict_summary: dict[ConstraintType, int]
    conflict: ConstraintConflict | None = None

    @model_validator(mode="after")
    def _statistics_consistent(self) -> ConstraintFilterResult:
        if self.feasible_count != len(self.feasible_outfits):
            raise ValueError("feasible_count does not match feasible_outfits")
        if self.rejected_count != len(self.rejected_outfits):
            raise ValueError("rejected_count does not match rejected_outfits")
        if self.total_candidates != self.feasible_count + self.rejected_count:
            raise ValueError("total_candidates must equal feasible_count + rejected_count")
        if self.compatibility_calls_saved != self.rejected_count:
            raise ValueError("compatibility_calls_saved must equal rejected_count")

        ids = [o.outfit_id for o in self.feasible_outfits] + [
            o.outfit_id for o in self.rejected_outfits
        ]
        repeated = sorted(i for i, n in Counter(ids).items() if n > 1)
        if repeated:
            raise ValueError(f"duplicate outfit_id in result: {repeated}")

        if set(self.conflict_summary) != set(ConstraintType):
            raise ValueError("conflict_summary must contain every ConstraintType exactly once")
        if self.conflict_summary != _summarise(self.rejected_outfits):
            raise ValueError("conflict_summary does not match the rejection reasons")

        nothing_feasible = self.total_candidates > 0 and self.feasible_count == 0
        if nothing_feasible and self.conflict is None:
            raise ValueError("conflict is required when no candidate outfit is feasible")
        if not nothing_feasible and self.conflict is not None:
            raise ValueError("conflict must be null unless all candidate outfits were rejected")
        if self.conflict is not None:
            if self.conflict.violated_constraints != _violated(self.conflict_summary):
                raise ValueError("conflict.violated_constraints does not match conflict_summary")
            expected = _min_budget_only_total(self.rejected_outfits)
            if self.conflict.min_budget_to_make_feasible != expected:
                raise ValueError(
                    f"conflict.min_budget_to_make_feasible must be {expected} (lowest total of "
                    f"outfits rejected only by BUDGET), got "
                    f"{self.conflict.min_budget_to_make_feasible}"
                )
        return self


def _summarise(rejected: list[RejectedOutfit]) -> dict[ConstraintType, int]:
    summary = {constraint: 0 for constraint in ConstraintType}
    for outfit in rejected:
        for reason in outfit.reasons:
            summary[reason.constraint] += 1
    return summary


def _violated(summary: dict[ConstraintType, int]) -> list[ConstraintType]:
    return [constraint for constraint in ConstraintType if summary[constraint] > 0]


def _min_budget_only_total(rejected: list[RejectedOutfit]) -> Decimal | None:
    """Lowest purchase total among outfits whose only violation is BUDGET, else None."""
    totals = [
        outfit.total_purchase_cost
        for outfit in rejected
        if [r.constraint for r in outfit.reasons] == [ConstraintType.BUDGET]
    ]
    return min(totals) if totals else None


def _purchase_items(outfit: CandidateOutfit) -> list[Product]:
    return [item for item in outfit.items if not item.is_user_owned]


def purchase_total(outfit: CandidateOutfit) -> Decimal:
    """TotalPrice(o): sum of purchase item prices; user-owned items are excluded."""
    return sum((item.price for item in _purchase_items(outfit)), Decimal("0"))


def _normalise_size(size: str) -> str:
    return size.strip().casefold()


def _check_unique_ids(candidates: list[CandidateOutfit]) -> None:
    counts = Counter(outfit.outfit_id for outfit in candidates)
    repeated = sorted(i for i, n in counts.items() if n > 1)
    if repeated:
        raise ValueError(f"candidate outfit_id values must be unique, duplicated: {repeated}")


def _check_currency(request: TripShoppingRequest, outfit: CandidateOutfit) -> None:
    for item in _purchase_items(outfit):
        if item.currency != request.currency:
            raise CurrencyMismatchError(
                f"outfit {outfit.outfit_id!r}: product {item.product_id!r} is priced in "
                f"{item.currency} but the request budget is in {request.currency}; "
                f"prices must be normalised to the budget currency before Stage 10"
            )


def check_outfit(
    request: TripShoppingRequest, outfit: CandidateOutfit
) -> tuple[list[RejectionReason], Decimal]:
    """Evaluate every hard constraint for one outfit.

    Returns all violations (empty if feasible) and the purchase total. All constraints are
    checked even after the first failure. Raises CurrencyMismatchError on currency conflicts.
    """
    _check_currency(request, outfit)
    purchases = _purchase_items(outfit)
    total = purchase_total(outfit)
    reasons: list[RejectionReason] = []

    if total > request.budget:
        reasons.append(
            RejectionReason(
                constraint=ConstraintType.BUDGET,
                message=f"purchase total {total} {request.currency} exceeds budget "
                f"{request.budget} {request.currency}",
            )
        )

    unavailable = [item.product_id for item in purchases if not item.availability]
    if unavailable:
        reasons.append(
            RejectionReason(
                constraint=ConstraintType.AVAILABILITY,
                message=f"purchase item(s) not available: {unavailable}",
                product_ids=unavailable,
            )
        )

    # PROVISIONAL: a single user_size is matched against every purchase item, because the
    # proposal defines no per-category size field. Products without reliable size data are
    # skipped rather than treated as a mismatch; with reliable data, an empty available_sizes
    # list means no size is in stock.
    if request.user_size is not None:
        wanted = _normalise_size(request.user_size)
        size_misses = [
            item.product_id
            for item in purchases
            if item.size_data_reliable
            and wanted not in {_normalise_size(s) for s in item.available_sizes}
        ]
        if size_misses:
            reasons.append(
                RejectionReason(
                    constraint=ConstraintType.SIZE,
                    message=f"size {request.user_size!r} not available for: {size_misses}",
                    product_ids=size_misses,
                )
            )

    return reasons, total


def _explain_conflict(
    request: TripShoppingRequest,
    rejected: list[RejectedOutfit],
    summary: dict[ConstraintType, int],
) -> ConstraintConflict:
    min_budget = _min_budget_only_total(rejected)
    violated = _violated(summary)

    parts = [
        f"No candidate outfit satisfies all hard constraints ({len(rejected)} rejected).",
        "Violations: " + ", ".join(f"{c.value} in {summary[c]} outfit(s)" for c in violated) + ".",
        f"Current budget is {request.budget} {request.currency}.",
    ]
    if min_budget is not None:
        parts.append(
            f"A budget of {min_budget} {request.currency} would make at least one outfit "
            f"feasible; the budget is not changed automatically."
        )
    return ConstraintConflict(
        violated_constraints=violated,
        budget=request.budget,
        currency=request.currency,
        min_budget_to_make_feasible=min_budget,
        message=" ".join(parts),
    )


def filter_candidate_outfits(
    request: TripShoppingRequest, candidates: list[CandidateOutfit]
) -> ConstraintFilterResult:
    """Stage 10: split candidates into feasible and rejected outfits, keeping input order.

    Inputs are not modified; feasible outfits are new FeasibleCandidateOutfit copies.
    Raises ValueError for duplicate outfit_id values and CurrencyMismatchError for
    currency conflicts.
    """
    _check_unique_ids(candidates)
    feasible: list[FeasibleCandidateOutfit] = []
    rejected: list[RejectedOutfit] = []

    for outfit in candidates:
        reasons, total = check_outfit(request, outfit)
        if reasons:
            rejected.append(
                RejectedOutfit(
                    outfit_id=outfit.outfit_id, reasons=reasons, total_purchase_cost=total
                )
            )
        else:
            feasible.append(FeasibleCandidateOutfit.model_validate(outfit.model_dump()))

    summary = _summarise(rejected)
    conflict = (
        _explain_conflict(request, rejected, summary) if candidates and not feasible else None
    )
    return ConstraintFilterResult(
        feasible_outfits=feasible,
        rejected_outfits=rejected,
        total_candidates=len(candidates),
        feasible_count=len(feasible),
        rejected_count=len(rejected),
        compatibility_calls_saved=len(rejected),
        conflict_summary=summary,
        conflict=conflict,
    )
