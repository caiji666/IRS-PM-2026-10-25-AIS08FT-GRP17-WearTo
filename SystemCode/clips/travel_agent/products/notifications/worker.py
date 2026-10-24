"""Authenticated pull worker; the Mac never exposes an inbound HTTP port."""

from pathlib import Path
from urllib.parse import urlsplit

import requests

from .lifecycle import Registry


class Worker:
    def __init__(self, url, token, registry, session=None, allow_local=False):
        parsed = urlsplit(url)
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("worker URL must be an origin without credentials")
        local = (
            allow_local
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.scheme == "http"
        )
        if parsed.scheme != "https" and not local:
            raise ValueError("HTTPS required for worker transport")
        if len(token) < 32:
            raise ValueError("strong worker token required")
        self.url, self.token, self.registry = url.rstrip("/"), token, registry
        self.session = session or requests.Session()

    def request(self, endpoint, method="POST", body=None):
        try:
            response = self.session.request(
                method,
                self.url + endpoint,
                json=body or {},
                headers={"Authorization": "Bearer " + self.token},
                timeout=(10, 30),
                allow_redirects=False,
            )
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"worker HTTP {response.status_code}")
            return response.json()
        except requests.RequestException:
            raise RuntimeError("worker network failure; credentials omitted") from None

    def once(self, apply=False):
        if not apply:
            return {"mode": "status_only", **self.request("/internal/status", "GET")}
        event = self.request("/internal/claim")["event"]
        if event is None:
            return {"status": "idle"}
        receipt = {"event_id": event["event_id"], "lease": event["lease"]}
        try:
            outcome = self.registry.cleanup(event)
        except Exception:
            self.request("/internal/nack", body=receipt)
            raise RuntimeError("cleanup failed; event retained for retry") from None
        self.request("/internal/ack", body=receipt)
        return outcome

    def close(self):
        self.registry.close()


def configured_worker(config, allow_local=False):
    root = Path(config["managed_root"]).resolve()
    registry = Registry(root, root / ".cache/ebay-lifecycle/registry.sqlite3")
    return Worker(config["service_url"], config["worker_token"], registry, allow_local=allow_local)
