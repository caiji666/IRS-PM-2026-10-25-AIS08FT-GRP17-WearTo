import numpy as np

from ..m5.models import cosine
from ..schemas import Availability, Category
from .models import (
    GroupedRankingReport,
    GroupedRankingRequest,
    RankedProduct,
    RankingReport,
    RankingRequest,
)


def _score_candidates(index, request: RankingRequest):
    # Validate compatibility even for an empty index or filtered-out vectors.
    for signal in (
        [request.context] if request.context is not None else []
    ) + request.social_references:
        if signal.model != index.model or len(signal.values) != index.dimension:
            raise ValueError("query embedding incompatible with index")
    cw = request.config.context_weight if request.context is not None else 0
    sw = request.config.social_weight if request.social_references else 0
    total = cw + sw
    cw, sw = cw / total, sw / total
    results = []
    for item in index.items:
        p = item.product
        if request.categories and p.category not in request.categories:
            continue
        if request.source_types and p.source_type not in request.source_types:
            continue
        if request.size and request.size.casefold() not in [x.casefold() for x in p.sizes]:
            continue
        if request.currency and (p.price is None or p.price.currency != request.currency):
            continue
        if request.max_price is not None and (
            p.price is None or p.price.amount > request.max_price
        ):
            continue
        if request.availability is not None:
            if p.availability != request.availability:
                continue
            if (
                request.availability != Availability.UNKNOWN
                and not p.evidence.availability_verified
            ):
                continue
        context = cosine(item.vector, request.context) if request.context is not None else None
        social = (
            max(cosine(item.vector, s) for s in request.social_references)
            if request.social_references
            else None
        )
        score = float(np.clip(cw * (context or 0) + sw * (social or 0), -1, 1))
        results.append(
            RankedProduct(
                item=item,
                score=score,
                context_similarity=context,
                social_similarity=social,
                context_weight=cw,
                social_weight=sw,
            )
        )
    results.sort(
        key=lambda r: (-r.score, r.item.product.source_type.value, r.item.product.product_id)
    )
    return results


def rank(index, request: RankingRequest):
    results = _score_candidates(index, request)
    return RankingReport(products=results[: request.limit], total_matches=len(results))


SUPPORTED_CATEGORIES = (Category.TOP, Category.BOTTOM, Category.FOOTWEAR, Category.OUTERWEAR)


def rank_by_category(index, request: GroupedRankingRequest):
    # Filter and score all candidates before grouping; never apply global limit first.
    results = _score_candidates(index, request)
    categories = list(dict.fromkeys(request.categories or SUPPORTED_CATEGORIES))
    groups = {category: [] for category in categories}
    excluded_unknown = 0
    for result in results:
        category = result.item.product.category
        if category in groups:
            groups[category].append(result)
        elif category == Category.UNKNOWN:
            excluded_unknown += 1
    counts = {category: len(items) for category, items in groups.items()}
    report = GroupedRankingReport(
        k_per_category=request.k_per_category,
        ranked_products={
            category: items[: request.k_per_category] for category, items in groups.items()
        },
        total_matches_per_category=counts,
        total_matches=sum(counts.values()),
        excluded_unknown_matches=excluded_unknown,
    )
    for category, count in counts.items():
        if count < request.k_per_category:
            report.warnings.append(
                f"{category.value}: only {count} eligible candidates for K={request.k_per_category}; no padding or duplicates."
            )
    if excluded_unknown:
        report.warnings.append(
            f"{excluded_unknown} UNKNOWN candidates excluded from supported-category groups."
        )
    return report
