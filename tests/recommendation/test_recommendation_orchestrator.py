"""Integration tests for Student5Orchestrator (Module 9) on MOCK candidate outfits.

The candidate outfits below are mock data for development and regression testing only. Their
compatibility_score values are arbitrary mock values, not Student 4 model output. Real
Student 4 → Student 5 integration lives in tests/test_real_integration.py (skipped until
Student 4 hands off its implementation).
"""

from decimal import Decimal

import pytest

from travel_agent.recommendation.student5.dialogue_state import StaleRecommendationSetError
from travel_agent.recommendation.student5.models import (
    CandidateOutfit,
    ConfirmedOutfit,
    PreviewMode,
    PreviewStatus,
    SessionStatus,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.recommendation.student5.orchestrator import (
    ConfirmationOutcome,
    RecommendationRun,
    RequirementChange,
    Student5Orchestrator,
)
from travel_agent.recommendation.student5.preview_adapter import (
    MockPreviewProvider,
    PreviewProviderError,
)
from travel_agent.recommendation.student5.ranking_engine import MissingCompatibilityScoreError

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


def mock_product(category, price, score=0.5, available=True):
    return {
        "product_id": f"p_{category.lower()}",
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
                mock_product("TOP", top, score),
                mock_product("BOTTOM", bottom, score),
                mock_product("FOOTWEAR", footwear, score, available=not unavailable),
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

    def test_state_records_round(self, first_run):
        state = first_run.state
        assert state.session_id == "session_001"
        assert (state.request_version, state.recommendation_round) == (1, 1)
        assert state.last_recommended_outfit_ids == ["o_best", "o_alt", "o_mid"]
        assert state.session_status is SessionStatus.AWAITING_CONFIRMATION

    def test_over_budget_and_out_of_stock_filtered(self, first_run):
        rejected = {
            r.outfit_id: [x.constraint.value for x in r.reasons]
            for r in first_run.filter_result.rejected_outfits
        }
        assert rejected == {"o_over_budget": ["BUDGET"], "o_out_of_stock": ["AVAILABILITY"]}
        assert "o_over_budget" not in ids(first_run.ranked_outfits)
        assert "o_out_of_stock" not in ids(first_run.ranked_outfits)

    def test_ranked_outfits_include_all_feasible(self, first_run):
        assert ids(first_run.ranked_outfits) == ["o_best", "o_alt", "o_mid", "o_low"]

    def test_compatibility_score_used_in_final_score(self):
        # Equal product scores: with beta=1 the order follows CompatibilityScore; with beta=0
        # the order follows ProductScore and compatibility no longer decides.
        candidates = [
            mock_outfit("compatible", 0.9, score=0.1),
            mock_outfit("relevant", 0.1, score=0.9),
        ]
        by_compat = Student5Orchestrator(beta=1.0).run(
            request=make_request(), candidate_outfits=candidates
        )
        by_relevance = Student5Orchestrator(beta=0.0).run(
            request=make_request(), candidate_outfits=candidates
        )
        assert ids(by_compat.ranked_outfits) == ["compatible", "relevant"]
        assert ids(by_relevance.ranked_outfits) == ["relevant", "compatible"]
        top = by_compat.recommendation_set.recommendations[0]
        assert top.compatibility_score == 0.9
        assert top.final_outfit_score == pytest.approx(0.9)

    def test_formula_applied(self, first_run):
        best = first_run.recommendation_set.recommendations[0]
        assert best.final_outfit_score == pytest.approx(0.5 * 0.9 + 0.5 * 0.5)
        assert best.total_purchase_cost == Decimal("150.00")

    def test_no_feasible_outfit_reports_conflict(self, orchestrator):
        candidates = [mock_outfit("too_expensive", 0.8, ("400.00", "100.00", "100.00"))]
        run = orchestrator.run(request=make_request(), candidate_outfits=candidates)
        assert run.recommendation_set is None
        assert run.ranked_outfits == []
        assert run.conflict is not None
        assert run.conflict.min_budget_to_make_feasible == Decimal("600.00")
        assert run.state.recommendation_round == 0  # nothing recorded

    def test_empty_candidates(self, orchestrator):
        run = orchestrator.run(request=make_request(), candidate_outfits=[])
        assert run.recommendation_set is None and run.conflict is None

    def test_missing_compatibility_score_not_hidden(self, orchestrator):
        candidates = [
            CandidateOutfit.model_validate(
                {**mock_outfit("unscored", 0.5).model_dump(), "compatibility_score": None}
            )
        ]
        with pytest.raises(MissingCompatibilityScoreError, match="unscored"):
            orchestrator.run(request=make_request(), candidate_outfits=candidates)

    def test_invalid_beta_rejected_at_construction(self):
        with pytest.raises(ValueError, match="beta"):
            Student5Orchestrator(beta=1.5)

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
    def test_modify_requirements_updates_state_and_plans_rerun(self, orchestrator, first_run):
        change = orchestrator.modify_requirements(first_run.state, patch(set={"budget": 170}))
        assert isinstance(change, RequirementChange)
        assert change.state.request_version == 2
        assert change.state.current_request.budget == Decimal("170")
        assert change.state.last_recommended_outfit_ids == []
        assert change.update.changed_fields == ["budget"]
        assert change.rerun_plan.affected_stages == [10, 11, 12]
        assert change.rerun_plan.rerun_from_stage == 10

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

    def test_no_change_gives_no_rerun_plan(self, orchestrator, first_run):
        change = orchestrator.modify_requirements(first_run.state, patch(set={"budget": 500}))
        assert change.update.changed is False
        assert change.rerun_plan is None
        assert change.state == first_run.state

    def test_confirm_outfit(self, orchestrator, first_run):
        outcome = orchestrator.confirm(first_run.state, first_run.recommendation_set, "o_alt")
        assert isinstance(outcome, ConfirmationOutcome)
        assert outcome.state.session_status is SessionStatus.CONFIRMED
        assert outcome.state.selected_outfit_id == "o_alt"
        assert isinstance(outcome.confirmed_outfit, ConfirmedOutfit)
        assert outcome.confirmed_outfit.recommendation.outfit_id == "o_alt"

    def test_explain_outfit(self, orchestrator, first_run):
        explanation = orchestrator.explain(first_run.state, first_run.recommendation_set, "o_mid")
        assert explanation.display_position == 3
        assert explanation.within_budget is True


# ---------------------------------------------------------------------------
# Stage 14
# ---------------------------------------------------------------------------


class TestPreview:
    def confirmed(self, orchestrator, first_run):
        return orchestrator.confirm(first_run.state, first_run.recommendation_set, "o_best")

    def test_mock_preview_after_confirmation(self, first_run):
        provider = MockPreviewProvider()
        orchestrator = Student5Orchestrator(beta=0.5, preview_provider=provider)
        outcome = self.confirmed(orchestrator, first_run)
        result = orchestrator.generate_preview(outcome.confirmed_outfit, PreviewMode.FLAT_LAY)
        assert result.status is PreviewStatus.SUCCESS
        assert result.selected_outfit_id == "o_best"
        assert provider.calls == [PreviewMode.FLAT_LAY]

    def test_portrait_preview_with_consent(self, first_run):
        orchestrator = Student5Orchestrator(beta=0.5, preview_provider=MockPreviewProvider())
        outcome = self.confirmed(orchestrator, first_run)
        result = orchestrator.generate_preview(
            outcome.confirmed_outfit,
            PreviewMode.PORTRAIT,
            user_portrait="uploads/me.png",
            portrait_consent=True,
        )
        assert result.delivered_mode is PreviewMode.PORTRAIT

    def test_preview_failure_keeps_confirmation(self, first_run):
        orchestrator = Student5Orchestrator(
            beta=0.5, preview_provider=MockPreviewProvider(failure=PreviewProviderError("down"))
        )
        outcome = self.confirmed(orchestrator, first_run)
        state_before = outcome.state.model_dump()
        confirmed_before = outcome.confirmed_outfit.model_dump()
        result = orchestrator.generate_preview(outcome.confirmed_outfit, PreviewMode.MANNEQUIN)
        assert result.status is PreviewStatus.FALLBACK
        assert result.delivered_mode is PreviewMode.PRODUCT_CARDS
        assert outcome.state.model_dump() == state_before
        assert outcome.state.session_status is SessionStatus.CONFIRMED
        assert outcome.confirmed_outfit.model_dump() == confirmed_before

    def test_no_provider_falls_back(self, orchestrator, first_run):
        outcome = self.confirmed(orchestrator, first_run)
        result = orchestrator.generate_preview(outcome.confirmed_outfit, PreviewMode.FLAT_LAY)
        assert result.status is PreviewStatus.FALLBACK

    def test_product_cards_needs_no_provider(self, orchestrator, first_run):
        outcome = self.confirmed(orchestrator, first_run)
        result = orchestrator.generate_preview(outcome.confirmed_outfit, PreviewMode.PRODUCT_CARDS)
        assert result.status is PreviewStatus.SUCCESS


# ---------------------------------------------------------------------------
# Inputs are never modified
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_full_flow_does_not_modify_inputs(self):
        provider = MockPreviewProvider()
        orchestrator = Student5Orchestrator(beta=0.5, preview_provider=provider)
        request = make_request()
        candidates = mock_candidates()
        snapshot = (request.model_dump(), [c.model_dump() for c in candidates], list(candidates))

        run = orchestrator.run(request=request, candidate_outfits=candidates)
        state_1 = run.state.model_dump()
        rs_1 = run.recommendation_set.model_dump()

        change = orchestrator.modify_requirements(run.state, patch(add={"style": ["casual"]}))
        rerun = orchestrator.run(
            request=change.state.current_request, candidate_outfits=candidates, state=change.state
        )
        change_state = change.state.model_dump()
        rerun_rs = rerun.recommendation_set.model_dump()
        outcome = orchestrator.confirm(rerun.state, rerun.recommendation_set, "o_alt")
        rerun_state = rerun.state.model_dump()
        confirmed = outcome.confirmed_outfit.model_dump()
        orchestrator.explain(outcome.state, rerun.recommendation_set, "o_best")
        orchestrator.generate_preview(outcome.confirmed_outfit, PreviewMode.FLAT_LAY)

        assert (request.model_dump(), [c.model_dump() for c in candidates], candidates) == snapshot
        assert run.state.model_dump() == state_1
        assert run.recommendation_set.model_dump() == rs_1
        assert change.state.model_dump() == change_state
        assert rerun.recommendation_set.model_dump() == rerun_rs
        assert rerun.state.model_dump() == rerun_state
        assert outcome.confirmed_outfit.model_dump() == confirmed

    def test_results_are_frozen(self, first_run):
        with pytest.raises(AttributeError):
            first_run.state = None
