"""Integration tests for Student5Orchestrator (Module 9) on MOCK candidate outfits.

The candidate outfits below are mock data for development and regression testing only. Their
compatibility_score values are arbitrary mock values, not Student 4 model output. Real
Student 4 → Student 5 integration lives in tests/test_real_integration.py (skipped until
Student 4 hands off its implementation).
"""

import pytest

from travel_agent.recommendation.student5.dialogue_state import StaleRecommendationSetError
from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    PreviewDeliveryMode,
    PreviewMode,
    PreviewStatus,
    SessionStatus,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.recommendation.student5.orchestrator import (
    RecommendationRun,
    Student5Orchestrator,
)
from travel_agent.recommendation.student5.preview_adapter import (
    MockPreviewProvider,
    PreviewProviderError,
)

# ---------------------------------------------------------------------------
# Mock fixtures (not real data)
# ---------------------------------------------------------------------------


def make_request(**overrides):
    data = {
        "destination": "Seoul",
        "start_date": "2026-12-20",
        "end_date": "2026-12-27",
        "activities": ["sightseeing"],
        "budget": 500,
        "currency": "SGD",
    }
    data.update(overrides)
    return TripShoppingRequest.model_validate(data)


def mock_product(category, price, score=0.5, available=True, outfit_id="outfit"):
    return {
        "product_id": f"{outfit_id}_{category.lower()}",
        "source": "mock",
        "title": f"Mock {category}",
        "description": None,
        "product_url": f"https://shop.example.com/{category.lower()}",
        "image_url": None,
        "price": price,
        "currency": "SGD",
        "availability": available,
        "category": category,
        "product_score": score,
    }


def mock_outfit(
    outfit_id, compat, prices=("50.00", "50.00", "50.00"), score=0.5, unavailable=False
):
    top, bottom, footwear = prices
    return CandidateOutfit.model_validate(
        {
            "outfit_id": outfit_id,
            "compatibility_score": compat,
            "items": [
                mock_product("TOP", top, score, outfit_id=outfit_id),
                mock_product("BOTTOM", bottom, score, outfit_id=outfit_id),
                mock_product(
                    "FOOTWEAR",
                    footwear,
                    score,
                    available=not unavailable,
                    outfit_id=outfit_id,
                ),
            ],
        }
    )


def mock_candidates():
    """MOCK candidates: compatibility scores are arbitrary test values."""
    return [
        mock_outfit("o_low", 0.30, ("60.00", "60.00", "60.00")),  # 180
        mock_outfit("o_best", 0.90, ("50.00", "50.00", "50.00")),  # 150
        mock_outfit("o_over_budget", 0.99, ("700.00", "100.00", "100.00")),
        mock_outfit("o_mid", 0.60, ("80.00", "60.00", "60.00")),  # 200
        mock_outfit("o_out_of_stock", 0.95, unavailable=True),
        mock_outfit("o_alt", 0.70, ("60.00", "50.00", "50.00")),  # 160
    ]


def ids(items):
    return [i.outfit_id for i in items]


@pytest.fixture
def orchestrator():
    return Student5Orchestrator(beta=0.5, session_id="session_001")


@pytest.fixture
def first_run(orchestrator):
    return orchestrator.run(request=make_request(), candidate_outfits=mock_candidates())


# ---------------------------------------------------------------------------
# Stages 10 → 12
# ---------------------------------------------------------------------------


class TestRecommendationRun:
    def test_top_three(self, first_run):
        assert isinstance(first_run, RecommendationRun)
        rs = first_run.recommendation_set
        assert [r.outfit_id for r in rs.recommendations] == ["o_best", "o_alt", "o_mid"]
        assert [r.display_position for r in rs.recommendations] == [1, 2, 3]
        assert rs.recommendation_round == 1

    def test_state_must_match_request(self, first_run, orchestrator):
        with pytest.raises(ValueError, match="current_request"):
            orchestrator.run(
                request=make_request(budget=100),
                candidate_outfits=mock_candidates(),
                state=first_run.state,
            )


# ---------------------------------------------------------------------------
# Stage 13
# ---------------------------------------------------------------------------


def patch(**operations):
    return TripShoppingRequestPatch.model_validate({"intent": "MODIFY_REQUIREMENTS", **operations})


class TestUserDecision:
    def test_rerun_after_modification(self, orchestrator, first_run):
        change = orchestrator.modify_requirements(first_run.state, patch(set={"budget": 170}))
        second = orchestrator.run(
            request=change.state.current_request,
            candidate_outfits=mock_candidates(),
            state=change.state,
        )
        rs = second.recommendation_set
        assert rs.recommendation_round == 2
        assert [r.outfit_id for r in rs.recommendations] == ["o_best", "o_alt"]
        assert second.state.last_recommended_outfit_ids == ["o_best", "o_alt"]
        with pytest.raises(StaleRecommendationSetError):
            orchestrator.confirm(second.state, first_run.recommendation_set, "o_best")


# ---------------------------------------------------------------------------
# Stage 14
# ---------------------------------------------------------------------------


class TestPreview:
    def confirmed(self, orchestrator, first_run):
        return orchestrator.confirm(first_run.state, first_run.recommendation_set, "o_best")

    def test_preview_failure_keeps_confirmation(self, first_run):
        orchestrator = Student5Orchestrator(
            beta=0.5, preview_provider=MockPreviewProvider(failure=PreviewProviderError("down"))
        )
        outcome = self.confirmed(orchestrator, first_run)
        state_before = outcome.state.model_dump()
        confirmed_before = outcome.confirmed_outfit.model_dump()
        result = orchestrator.generate_preview(outcome.confirmed_outfit, PreviewMode.FLAT_LAY)
        assert result.status is PreviewStatus.FALLBACK
        assert result.delivered_mode is PreviewDeliveryMode.PRODUCT_CARDS
        assert outcome.state.model_dump() == state_before
        assert outcome.state.session_status is SessionStatus.CONFIRMED
        assert outcome.confirmed_outfit.model_dump() == confirmed_before


# ---------------------------------------------------------------------------
# Inputs are never modified
# ---------------------------------------------------------------------------
