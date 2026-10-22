import http.client
import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer

import pytest

from travel_agent.schemas import ModuleResult
from travel_agent.web import __main__ as web


def _proposal():
    return {
        "patch": {"intent": "MODIFY_REQUIREMENTS", "set": {"destination": "Tokyo"}},
        "updated_request": {"destination": "Tokyo"},
        "changed_fields": ["destination"],
        "rerun_plan": {
            "changed_fields": ["destination"],
            "rerun_from_stage": 2,
            "affected_stages": list(range(2, 13)),
            "reason": "private detailed reason",
        },
        "student1_state": {"private": "state"},
    }


def test_pending_refinement_ids_are_opaque_and_public_plan_is_sanitized():
    from travel_agent.web.refinement_jobs import PendingRefinementStore

    store = PendingRefinementStore()
    pending = store.create_pending("session-a", 4, _proposal())
    public = store.public_plan(pending)

    assert len(pending.pending_change_id) == 32
    assert pending.pending_change_id.isalnum()
    assert public == {
        "status": "confirmation_required",
        "pending_change_id": pending.pending_change_id,
        "changed_fields": ["destination"],
        "rerun_from_stage": 2,
        "affected_stages": list(range(2, 13)),
        "message": "This change requires weather, products and outfits to be updated.",
    }
    assert "student1_state" not in public
    assert "patch" not in public


def test_pending_refinement_replaces_previous_change_for_same_session():
    from travel_agent.web.refinement_jobs import PendingRefinementError, PendingRefinementStore

    store = PendingRefinementStore()
    old = store.create_pending("session-a", 4, _proposal())
    new = store.create_pending("session-a", 4, _proposal())

    with pytest.raises(PendingRefinementError, match="unknown or replaced"):
        store.consume_pending(old.pending_change_id, "session-a", 4, "job-old")
    consumed = store.consume_pending(new.pending_change_id, "session-a", 4, "job-new")
    assert consumed.job_id == "job-new"


@pytest.mark.parametrize(
    ("session_id", "request_version", "message"),
    [("session-b", 4, "different session"), ("session-a", 5, "stale")],
)
def test_pending_refinement_rejects_mismatch(session_id, request_version, message):
    from travel_agent.web.refinement_jobs import PendingRefinementError, PendingRefinementStore

    store = PendingRefinementStore()
    pending = store.create_pending("session-a", 4, _proposal())
    with pytest.raises(PendingRefinementError, match=message):
        store.consume_pending(pending.pending_change_id, session_id, request_version, "job")


def test_pending_refinement_expires_and_is_consumed_once():
    from travel_agent.web.refinement_jobs import PendingRefinementError, PendingRefinementStore

    now = datetime(2026, 10, 22, tzinfo=timezone.utc)
    store = PendingRefinementStore(ttl=timedelta(minutes=5), now=lambda: now)
    expired = store.create_pending("session-a", 4, _proposal())
    store._now = lambda: now + timedelta(minutes=6)
    with pytest.raises(PendingRefinementError, match="expired"):
        store.consume_pending(expired.pending_change_id, "session-a", 4, "job")

    store._now = lambda: now
    pending = store.create_pending("session-a", 4, _proposal())
    first = store.consume_pending(pending.pending_change_id, "session-a", 4, "job-1")
    second = store.consume_pending(pending.pending_change_id, "session-a", 4, "job-2")
    assert first.job_id == second.job_id == "job-1"


def test_pending_refinement_can_release_a_job_that_never_started():
    from travel_agent.web.refinement_jobs import PendingRefinementStore

    store = PendingRefinementStore()
    pending = store.create_pending("session-a", 4, _proposal())
    store.consume_pending(pending.pending_change_id, "session-a", 4, "job-1")
    store.release_job(pending.pending_change_id, "job-1")

    retried = store.consume_pending(pending.pending_change_id, "session-a", 4, "job-2")
    assert retried.job_id == "job-2"


def test_http_upstream_refinement_requires_confirmation_and_confirmation_is_idempotent(
    tmp_path, monkeypatch
):
    from travel_agent.web.refinement_jobs import PendingRefinementStore

    request = {
        "destination": "Seoul",
        "start_date": "2026-11-01",
        "end_date": "2026-11-05",
        "activities": ["sightseeing"],
        "budget": 500,
        "currency": "USD",
        "style": ["minimalist"],
        "excluded_items": [],
        "user_size": "M",
    }
    session_id = "a" * 32
    state = {
        "session_id": "http-test",
        "request_version": 4,
        "current_request": request,
        "recommendation_round": 2,
        "last_recommended_outfit_ids": ["OLD"],
        "selected_outfit_id": None,
        "session_status": "AWAITING_CONFIRMATION",
    }
    monkeypatch.setattr(web, "ROOT", tmp_path)
    monkeypatch.setattr(web, "JOBS", tmp_path / "jobs")
    web.JOBS.mkdir()
    monkeypatch.setattr(
        web, "STUDENT5_SESSIONS", {session_id: {"request": request, "state": state}}
    )
    monkeypatch.setattr(web, "STUDENT5_REFINEMENTS", PendingRefinementStore())

    def fake_understand(payload, **kwargs):
        proposed = {**payload["state"]["current_request"], "destination": "Tokyo"}
        return ModuleResult(
            status="needs_input",
            reason="Human extraction review required",
            data={
                "state": {**payload["state"], "pending_request": proposed},
                "proposed_request": proposed,
                "questions": [{"field": "extraction_review", "question": "Review"}],
            },
        )

    monkeypatch.setattr("travel_agent.understanding.service.run", fake_understand)
    started = []

    def fake_worker(job, selected_session, pending):
        started.append((job, selected_session, pending.pending_change_id))
        web.update(job, status="running", stage="weather")
        web.ACTIVE.release()

    monkeypatch.setattr(web, "execute_student5_refinement", fake_worker)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)

    def post(path, payload):
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={
                "Host": f"127.0.0.1:{server.server_port}",
                "X-Trip-UI": "1",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    try:
        status, plan = post(
            "/api/student5/refine",
            {"id": session_id, "message": "Change my destination to Tokyo"},
        )
        assert status == 200
        assert plan["status"] == "confirmation_required"
        assert plan["changed_fields"] == ["destination"]
        payload = {"id": session_id, "pending_change_id": plan["pending_change_id"]}
        first_status, first = post("/api/student5/refine/confirm", payload)
        second_status, second = post("/api/student5/refine/confirm", payload)
        assert first_status == second_status == 202
        assert first["id"] == second["id"]
        assert len(started) == 1
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        if web.ACTIVE.locked():
            web.ACTIVE.release()
