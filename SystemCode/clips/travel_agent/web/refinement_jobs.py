"""Short-lived, server-side refinement approvals bound to one session version."""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


class PendingRefinementError(ValueError):
    """A pending refinement cannot be used for this session state."""


@dataclass
class PendingRefinement:
    pending_change_id: str
    session_id: str
    request_version: int
    proposal: dict
    expires_at: datetime
    job_id: str | None = None


class PendingRefinementStore:
    def __init__(
        self,
        *,
        ttl: timedelta = timedelta(minutes=15),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._ttl = ttl
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._by_id: dict[str, PendingRefinement] = {}
        self._by_session: dict[str, str] = {}

    def create_pending(
        self, session_id: str, request_version: int, proposal: dict
    ) -> PendingRefinement:
        with self._lock:
            previous_id = self._by_session.get(session_id)
            if previous_id:
                self._by_id.pop(previous_id, None)
            pending = PendingRefinement(
                pending_change_id=secrets.token_hex(16),
                session_id=session_id,
                request_version=request_version,
                proposal=proposal,
                expires_at=self._now() + self._ttl,
            )
            self._by_id[pending.pending_change_id] = pending
            self._by_session[session_id] = pending.pending_change_id
            return pending

    def consume_pending(
        self,
        pending_change_id: str,
        session_id: str,
        request_version: int,
        job_id: str,
    ) -> PendingRefinement:
        with self._lock:
            pending = self._by_id.get(pending_change_id)
            if pending is None:
                raise PendingRefinementError("Pending change is unknown or replaced.")
            if pending.session_id != session_id:
                raise PendingRefinementError("Pending change belongs to a different session.")
            if pending.request_version != request_version:
                raise PendingRefinementError("Pending change is stale for this request version.")
            if self._now() >= pending.expires_at:
                self._by_id.pop(pending_change_id, None)
                if self._by_session.get(session_id) == pending_change_id:
                    self._by_session.pop(session_id, None)
                raise PendingRefinementError("Pending change has expired.")
            if pending.job_id is None:
                pending.job_id = job_id
            return pending

    def release_job(self, pending_change_id: str, job_id: str) -> None:
        """Make a reserved change retryable when its worker could not be started."""
        with self._lock:
            pending = self._by_id.get(pending_change_id)
            if pending is not None and pending.job_id == job_id:
                pending.job_id = None

    @staticmethod
    def public_plan(pending: PendingRefinement) -> dict:
        rerun = pending.proposal["rerun_plan"]
        return {
            "status": "confirmation_required",
            "pending_change_id": pending.pending_change_id,
            "changed_fields": list(pending.proposal["changed_fields"]),
            "rerun_from_stage": rerun["rerun_from_stage"],
            "affected_stages": list(rerun["affected_stages"]),
            "message": "This change requires weather, products and outfits to be updated.",
        }
