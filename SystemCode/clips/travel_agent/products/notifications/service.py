"""WSGI service: public eBay callback; authenticated private worker endpoints."""

import hmac
import ipaddress
import json
import logging
import os
import re
import sqlite3
from urllib.parse import parse_qs, urlsplit

from ..retrieval.ebay.client import BrowseClient, EbayError
from .security import PublicKeys, Verifier, challenge
from .store import Events

LOG = logging.getLogger(__name__)
MAX_BODY = 256_000


class Application:
    def __init__(self, endpoint, verification_token, worker_tokens, events, verifier):
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("endpoint must be an exact public HTTPS URL")
        host = parsed.hostname
        if host == "localhost" or host.endswith(".local"):
            raise ValueError("public endpoint required")
        try:
            if not ipaddress.ip_address(host).is_global:
                raise ValueError("public endpoint required")
        except ValueError as exc:
            if str(exc) == "public endpoint required":
                raise
        if parsed.path == "/healthz" or parsed.path.startswith("/internal/"):
            raise ValueError("callback conflicts with internal routes")
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,80}", verification_token):
            raise ValueError("invalid verification token")
        if not worker_tokens or any(len(t) < 32 for t in worker_tokens.values()):
            raise ValueError("strong per-consumer worker tokens required")
        if set(worker_tokens) != set(events.consumers):
            raise ValueError("queue consumers must match configured worker credentials")
        self.endpoint, self.callback = endpoint, parsed.path or "/"
        self.token, self.workers, self.events, self.verifier = (
            verification_token,
            worker_tokens,
            events,
            verifier,
        )

    def __call__(self, env, start):
        def reply(status, body):
            data = json.dumps(body).encode() if body is not None else b""
            start(
                status,
                [
                    ("Content-Type", "application/json"),
                    ("Cache-Control", "no-store"),
                    ("Content-Length", str(len(data))),
                ],
            )
            return [data]

        try:
            method, path = env.get("REQUEST_METHOD"), env.get("PATH_INFO")
            if method == "GET" and path == "/healthz":
                self.events.health()
                return reply("200 OK", {"status": "up"})
            if path == self.callback and method == "GET":
                codes = parse_qs(env.get("QUERY_STRING", "")).get("challenge_code", [])
                if len(codes) != 1:
                    raise ValueError("one challenge code required")
                return reply(
                    "200 OK", {"challengeResponse": challenge(codes[0], self.token, self.endpoint)}
                )
            worker = None
            if path.startswith("/internal/"):
                supplied = env.get("HTTP_AUTHORIZATION", "")
                worker = next(
                    (
                        c
                        for c, t in self.workers.items()
                        if hmac.compare_digest(supplied, "Bearer " + t)
                    ),
                    None,
                )
                if worker is None:
                    return reply("401 Unauthorized", {"error": "unauthorized"})
                if method == "GET" and path == "/internal/status":
                    return reply("200 OK", self.events.health())
            if method != "POST" or path not in {
                self.callback,
                "/internal/claim",
                "/internal/ack",
                "/internal/nack",
            }:
                return reply("404 Not Found", {"error": "not_found"})
            size = int(env.get("CONTENT_LENGTH") or 0)
            if not 0 < size <= MAX_BODY:
                return reply("413 Payload Too Large", {"error": "body_limit"})
            if env.get("CONTENT_TYPE", "").split(";")[0].lower() != "application/json":
                return reply("415 Unsupported Media Type", {"error": "json_required"})
            raw = env["wsgi.input"].read(size)
            if len(raw) != size:
                raise ValueError("truncated body")
            if path == self.callback:
                try:
                    message = self.verifier.verify(raw, env.get("HTTP_X_EBAY_SIGNATURE", ""))
                    self.events.put(message)
                except (ValueError, KeyError, TypeError):
                    LOG.warning("notification_rejected")
                    return reply("412 Precondition Failed", {"error": "invalid_notification"})
                LOG.info("notification_persisted")
                return reply("204 No Content", None)
            if path == "/internal/claim":
                return reply("200 OK", {"event": self.events.claim(worker)})
            body = json.loads(raw)
            self.events.finish(worker, body["event_id"], body["lease"], path == "/internal/ack")
            return reply("200 OK", {"status": "recorded"})
        except (EbayError, sqlite3.Error, OSError):
            LOG.error("notification_dependency_failure")
            return reply("503 Service Unavailable", {"error": "temporary_failure"})
        except (ValueError, KeyError, TypeError):
            return reply("400 Bad Request", {"error": "invalid_request"})


def create_app():
    # Explicit factory: importing this module never contacts eBay or creates a database.
    workers = json.loads(os.environ["EBAY_WORKER_TOKENS_JSON"])
    events = Events(os.environ["EBAY_NOTIFICATION_DB"], os.environ["EBAY_QUEUE_KEY"], workers)
    client = BrowseClient(
        {
            "environment": os.environ.get("EBAY_ENVIRONMENT", "production"),
            "client_id": os.environ["EBAY_CLIENT_ID"],
            "client_secret": os.environ["EBAY_CLIENT_SECRET"],
        }
    )
    return Application(
        os.environ["EBAY_NOTIFICATION_ENDPOINT"],
        os.environ["EBAY_VERIFICATION_TOKEN"],
        workers,
        events,
        Verifier(PublicKeys(client)),
    )
