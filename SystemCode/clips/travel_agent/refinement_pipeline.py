"""Deterministic Student 1–5 coordinator for approved requirement changes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol


class RefinementPipelineError(RuntimeError):
    """A pipeline stage failed before a replacement session was ready."""


class Services(Protocol):
    understanding: Any
    inspiration: Any
    products: Any
    outfits: Any
    recommendation: Any


@dataclass(frozen=True)
class RefinementOutcome:
    request: dict
    checkpoints: dict
    student4: dict
    state: dict
    recommendation_set: dict
    filter_summary: dict


def _data(result: Any, stage: str) -> dict:
    if result.status != "ok":
        raise RefinementPipelineError(result.reason or f"{stage} failed")
    return deepcopy(result.data or {})


def _run(service: Any, payload: dict, stage: str) -> dict:
    return _data(service(payload), stage)


def run_refinement(
    request: dict,
    rerun_plan: dict,
    checkpoints: dict,
    progress: Any,
    services: Services,
) -> RefinementOutcome:
    """Run the minimum safe module boundary and return only a complete replacement."""
    original = deepcopy(checkpoints)
    working = deepcopy(checkpoints)
    start = int(rerun_plan["rerun_from_stage"])

    # Stage 3 is Student 1 hybrid reasoning, so it cannot reuse the old understanding output.
    required = ("understanding", "inspiration", "products", "student4")
    if any(key not in working for key in required):
        start = 2

    try:
        if start <= 3:
            progress(2, "Updating trip context and weather")
            understanding = _run(
                services.understanding, {"request": deepcopy(request)}, "understanding"
            )
            working = {"understanding": understanding}
        else:
            understanding = working["understanding"]

        if start <= 7:
            progress(5, "Refreshing style inspiration")
            inspiration = _run(
                services.inspiration,
                {"understanding": deepcopy(understanding)},
                "inspiration",
            )
            working = {"understanding": understanding, "inspiration": inspiration}
        else:
            inspiration = working["inspiration"]

        if start <= 10:
            progress(8, "Refreshing product candidates")
            products = _run(
                services.products,
                {
                    "understanding": deepcopy(understanding),
                    "inspiration": deepcopy(inspiration),
                },
                "products",
            )
            working = {
                "understanding": understanding,
                "inspiration": inspiration,
                "products": products,
            }
        else:
            products = working["products"]

        progress(11, "Scoring candidate outfits")
        student4 = _run(
            services.outfits,
            {
                "products": deepcopy(products),
                "understanding": deepcopy(understanding),
            },
            "outfits",
        )
        working["student4"] = student4

        progress(12, "Ranking the new Top-3")
        ranked = _run(
            services.recommendation,
            {
                "action": "recommend",
                "request": deepcopy(request),
                "student4": deepcopy(student4),
            },
            "recommendation",
        )
        return RefinementOutcome(
            request=deepcopy(request),
            checkpoints=working,
            student4=student4,
            state=ranked["state"],
            recommendation_set=ranked["recommendation_set"],
            filter_summary=ranked["filter_summary"],
        )
    except Exception:
        # The caller receives no partially updated checkpoint set.
        assert checkpoints == original
        raise
