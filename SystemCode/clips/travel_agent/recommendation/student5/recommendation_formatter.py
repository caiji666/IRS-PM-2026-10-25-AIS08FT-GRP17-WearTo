"""Stage 12 — build and present the RecommendationSet from ranked outfits.

Each Top-3 outfit becomes a RecommendationItem with its display position (1..n in ranked
order), existing (user-owned) items separated from purchase items, and a total purchase cost
covering purchase items only. Products are carried over with all their fields; outfit_id
values and scores are passed through unchanged.

The structured output is RecommendationSet itself (model_dump / model_dump_json).
format_recommendation_set_text renders the same data as deterministic plain text.

This module does not manage session state, confirmation, or requirement changes (Stage 13),
and does not generate recommendation explanations: those belong to the later Stage 13
REQUEST_EXPLANATION flow.
Function names, EmptyRecommendationError and the optional `currency` parameter are PROVISIONAL.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

from travel_agent.recommendation.student5.constraint_engine import CurrencyMismatchError
from travel_agent.recommendation.student5.models import (
    Product,
    RecommendationItem,
    RecommendationSet,
    ScoredOutfit,
)

MAX_RECOMMENDATIONS = 3


class EmptyRecommendationError(ValueError):
    """PROVISIONAL. No ranked outfit is available, so no valid RecommendationSet exists.

    The no-feasible-outfit case should be reported with Stage 10's ConstraintConflict instead.
    """


def format_recommendation_item(
    scored_outfit: ScoredOutfit, display_position: int, currency: str | None = None
) -> RecommendationItem:
    """Convert one ScoredOutfit into a RecommendationItem without modifying it.

    `currency` defaults to the purchase items' currency. Pass it explicitly (normally the
    request currency) when the outfit may consist of user-owned items only.
    """
    if not isinstance(scored_outfit, ScoredOutfit):
        raise TypeError(
            f"format_recommendation_item expects a ScoredOutfit, got {type(scored_outfit).__name__}"
        )
    outfit = scored_outfit.model_dump()
    existing = [item for item in outfit["items"] if item["is_user_owned"]]
    purchases = [item for item in outfit["items"] if not item["is_user_owned"]]

    if currency is None:
        if not purchases:
            raise ValueError(
                f"outfit {scored_outfit.outfit_id!r} has no purchase items; pass currency "
                f"explicitly"
            )
        currency = purchases[0]["currency"]
    for item in purchases:
        if item["currency"] != currency:
            raise CurrencyMismatchError(
                f"outfit {scored_outfit.outfit_id!r}: product {item['product_id']!r} is priced "
                f"in {item['currency']} but the recommendation currency is {currency}"
            )

    return RecommendationItem.model_validate(
        {
            "outfit_id": scored_outfit.outfit_id,
            "display_position": display_position,
            "existing_items": existing,
            "purchase_items": purchases,
            "total_purchase_cost": sum((item["price"] for item in purchases), Decimal("0")),
            "currency": currency,
            "compatibility_score": scored_outfit.compatibility_score,
            "outfit_product_relevance": scored_outfit.outfit_product_relevance,
            "final_outfit_score": scored_outfit.final_outfit_score,
        }
    )


def build_recommendation_set(
    ranked_outfits: list[ScoredOutfit],
    recommendation_round: int,
    currency: str | None = None,
) -> RecommendationSet:
    """Build the current RecommendationSet from already ranked (and truncated) outfits.

    Raises EmptyRecommendationError for an empty list, ValueError for more than 3 outfits
    (truncate with ranking_engine.select_top_outfits first) or duplicate outfit_id values.
    """
    if not ranked_outfits:
        raise EmptyRecommendationError("cannot build a RecommendationSet from no outfits")
    if len(ranked_outfits) > MAX_RECOMMENDATIONS:
        raise ValueError(
            f"at most {MAX_RECOMMENDATIONS} outfits allowed, got {len(ranked_outfits)}; "
            f"use select_top_outfits first"
        )
    repeated = sorted(i for i, n in Counter(o.outfit_id for o in ranked_outfits).items() if n > 1)
    if repeated:
        raise ValueError(f"outfit_id values must be unique, duplicated: {repeated}")

    items = [
        format_recommendation_item(outfit, position, currency)
        for position, outfit in enumerate(ranked_outfits, start=1)
    ]
    return RecommendationSet(recommendation_round=recommendation_round, recommendations=items)


NOT_AVAILABLE = "Not available"


def _format_product_line(product: Product) -> str:
    link = product.product_url or NOT_AVAILABLE
    image = product.image_url or NOT_AVAILABLE
    return (
        f"    - [{product.category.value}] {product.title} - {product.price} {product.currency}"
        f" | Link: {link} | Image: {image}"
    )


def _format_item_text(item: RecommendationItem) -> list[str]:
    lines = [f"#{item.display_position} outfit_id: {item.outfit_id}"]
    if item.existing_items:
        lines.append("  Existing items:")
        lines.extend(_format_product_line(p) for p in item.existing_items)
    if item.purchase_items:
        lines.append("  Items to purchase:")
        lines.extend(_format_product_line(p) for p in item.purchase_items)
    else:
        lines.append("  Items to purchase: none")
    lines.extend(
        [
            f"  Total purchase cost: {item.total_purchase_cost} {item.currency}",
            f"  CompatibilityScore: {item.compatibility_score:.4f}",
            f"  OutfitProductRelevance: {item.outfit_product_relevance:.4f}",
            f"  FinalOutfitScore: {item.final_outfit_score:.4f}",
        ]
    )
    return lines


def format_recommendation_set_text(recommendation_set: RecommendationSet) -> str:
    """PROVISIONAL. Render a RecommendationSet as deterministic plain text.

    Shows existing items (only when present) separately from items to purchase, each with
    category, title, price and currency, product link and image URL ("Not available" when
    missing), followed by the total purchase cost and the three stored scores (4 decimals).
    Presentation only: no explanations, no randomness, no ANSI codes; the input is not modified.
    """
    if not isinstance(recommendation_set, RecommendationSet):
        raise TypeError(
            f"format_recommendation_set_text expects a RecommendationSet, "
            f"got {type(recommendation_set).__name__}"
        )
    blocks = [f"Recommendation round {recommendation_set.recommendation_round}"]
    blocks.extend("\n".join(_format_item_text(item)) for item in recommendation_set.recommendations)
    return "\n\n".join(blocks)
