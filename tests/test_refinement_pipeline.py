from types import SimpleNamespace

import pytest


class Result:
    def __init__(self, status="ok", data=None, reason=None):
        self.status = status
        self.data = data or {}
        self.reason = reason

    def model_dump(self, mode="json"):
        return {"status": self.status, "data": self.data, "reason": self.reason}


def _services(calls, *, fail=None):
    def service(name, data):
        def execute(payload):
            calls.append(name)
            if fail == name:
                return Result("error", reason=f"{name} failed")
            return Result(data=data(payload) if callable(data) else data)

        return execute

    return SimpleNamespace(
        understanding=service(
            "understanding",
            lambda payload: {"trip_request": payload["request"], "ready_for_downstream": True},
        ),
        inspiration=service("inspiration", {"references": ["ref"]}),
        products=service("products", {"bundle": {"top_k_products": {}}}),
        outfits=service(
            "outfits",
            {
                "status": "ok",
                "source": "student4",
                "model_version": "test",
                "outfits": [{"outfit_id": "NEW"}],
            },
        ),
        recommendation=service(
            "recommendation",
            {
                "state": {"recommendation_round": 3},
                "recommendation_set": {"recommendations": [{"outfit_id": "NEW"}]},
                "filter_summary": {"feasible_count": 1},
            },
        ),
    )


def _plan(start=2):
    return {"rerun_from_stage": start, "affected_stages": list(range(start, 13))}


def test_refinement_pipeline_full_runs_without_checkpoints_and_reports_progress():
    from travel_agent.refinement_pipeline import run_refinement

    calls = []
    progress = []
    outcome = run_refinement(
        {"destination": "Tokyo"},
        _plan(2),
        {},
        lambda stage, message: progress.append((stage, message)),
        _services(calls),
    )

    assert calls == ["understanding", "inspiration", "products", "outfits", "recommendation"]
    assert [stage for stage, _ in progress] == [2, 5, 8, 11, 12]
    assert outcome.student4["outfits"][0]["outfit_id"] == "NEW"
    assert outcome.recommendation_set["recommendations"][0]["outfit_id"] == "NEW"
    assert set(outcome.checkpoints) == {"understanding", "inspiration", "products", "student4"}


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (3, ["understanding", "inspiration", "products", "outfits", "recommendation"]),
        (4, ["inspiration", "products", "outfits", "recommendation"]),
        (8, ["products", "outfits", "recommendation"]),
    ],
)
def test_refinement_pipeline_reuses_only_valid_upstream_checkpoints(start, expected):
    from travel_agent.refinement_pipeline import run_refinement

    calls = []
    checkpoints = {
        "understanding": {"trip_request": {"destination": "Seoul"}},
        "inspiration": {"references": ["old"]},
        "products": {"bundle": {"top_k_products": {}}},
        "student4": {"status": "ok", "outfits": [{"outfit_id": "OLD"}]},
    }
    run_refinement(
        {"destination": "Tokyo"}, _plan(start), checkpoints, lambda *_: None, _services(calls)
    )
    assert calls == expected


def test_refinement_pipeline_failure_returns_no_partial_outcome_and_keeps_input_unchanged():
    from travel_agent.refinement_pipeline import RefinementPipelineError, run_refinement

    calls = []
    checkpoints = {"understanding": {"old": True}}
    snapshot = {"understanding": {"old": True}}
    with pytest.raises(RefinementPipelineError, match="products failed"):
        run_refinement(
            {"destination": "Tokyo"},
            _plan(2),
            checkpoints,
            lambda *_: None,
            _services(calls, fail="products"),
        )
    assert checkpoints == snapshot
