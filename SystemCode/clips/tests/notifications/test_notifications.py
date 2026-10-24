import base64
import hashlib
import io
import json
import time

import pytest

pytest.importorskip("cryptography")
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from travel_agent.products.notifications.lifecycle import DeletedAccount, Registry, save_report
from travel_agent.products.notifications.security import PublicKeys, Verifier, challenge
from travel_agent.products.notifications.service import Application
from travel_agent.products.notifications.store import Events, database
from travel_agent.products.notifications.worker import Worker


@pytest.fixture
def signed():
    key = ec.generate_private_key(ec.SECP256R1())
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    record = {"key": public, "algorithm": "ECDSA", "digest": "SHA1"}
    message = {
        "metadata": {"topic": "MARKETPLACE_ACCOUNT_DELETION"},
        "notification": {
            "notificationId": "event-1",
            "publishAttemptCount": 1,
            "data": {"username": "seller-a", "userId": "user-1"},
        },
    }
    raw = json.dumps(message, separators=(",", ":")).encode()
    signature = key.sign(raw, ec.ECDSA(hashes.SHA1()))
    header = base64.b64encode(
        json.dumps(
            {
                "kid": "test-key",
                "alg": "ecdsa",
                "digest": "SHA1",
                "signature": base64.b64encode(signature).decode(),
            }
        ).encode()
    ).decode()
    return raw, header, Verifier(lambda kid: record), message


def invoke(app, method, path, body=b"", auth=None, signature="", query=""):
    env = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": query,
        "CONTENT_LENGTH": str(len(body)),
        "CONTENT_TYPE": "application/json",
        "wsgi.input": io.BytesIO(body),
        "HTTP_AUTHORIZATION": auth or "",
        "HTTP_X_EBAY_SIGNATURE": signature,
    }
    status = []
    raw = b"".join(app(env, lambda code, headers: status.append(code)))
    return int(status[0].split()[0]), json.loads(raw) if raw else None


@pytest.fixture
def receiver(tmp_path, signed):
    events = Events(tmp_path / "events.sqlite3", Fernet.generate_key().decode(), ["mac", "other"])
    app = Application(
        "https://example.com/ebay/account-deletion",
        "v" * 40,
        {"mac": "a" * 40, "other": "b" * 40},
        events,
        signed[2],
    )
    return app, events


def test_challenge_exact_endpoint_and_token(receiver):
    app, _ = receiver
    code, body = invoke(app, "GET", app.callback, query="challenge_code=abc")
    assert code == 200
    assert (
        body["challengeResponse"]
        == hashlib.sha256(("abc" + "v" * 40 + app.endpoint).encode()).hexdigest()
    )
    assert invoke(app, "GET", app.callback)[0] == 400
    assert invoke(app, "GET", app.callback, query="challenge_code=a&challenge_code=b")[0] == 400
    with pytest.raises(ValueError):
        challenge("", "v" * 40, app.endpoint)


def test_signature_tamper_bad_header_and_algorithm(signed):
    raw, header, verifier, message = signed
    assert verifier.verify(raw, header) == message
    assert verifier.verify(json.dumps(message, indent=2).encode(), header) == message
    for changed in (raw.replace(b"seller-a", b"seller-b"), b"{}"):
        with pytest.raises(ValueError):
            verifier.verify(changed, header)
    for invalid in ("", "!!!", base64.b64encode(b'{"alg":"rsa"}').decode()):
        with pytest.raises(ValueError):
            verifier.verify(raw, invalid)


def test_signed_receive_dedup_and_private_auth(receiver, signed):
    app, events = receiver
    raw, header, _, _ = signed
    assert invoke(app, "POST", app.callback, raw, signature=header)[0] == 204
    assert invoke(app, "POST", app.callback, raw, signature=header)[0] == 204
    assert events.health()["queue_counts"] == {"pending": 2}
    assert invoke(app, "GET", "/internal/status")[0] == 401
    assert invoke(app, "GET", "/healthz")[1] == {"status": "up"}
    assert invoke(app, "POST", app.callback, raw, signature="bad")[0] == 412
    assert b"seller-a" not in events.path.read_bytes()
    assert invoke(app, "POST", app.callback, b"")[0] == 413


def test_all_consumers_ack_before_payload_erasure(receiver, signed):
    _, events = receiver
    events.put(signed[3])
    event = events.claim("mac")
    assert events.claim("mac") is None
    with pytest.raises(ValueError):
        events.finish("other", event["event_id"], event["lease"], True)
    events.finish("mac", event["event_id"], event["lease"], True)
    with database(events.path) as db:
        assert db.execute("SELECT payload FROM events").fetchone()[0] is not None
    other = events.claim("other")
    events.finish("other", other["event_id"], other["lease"], True)
    with database(events.path) as db:
        assert db.execute("SELECT payload FROM events").fetchone()[0] is None
    events.put(signed[3])
    assert events.claim("mac") is None
    duplicate = json.loads(json.dumps(signed[3]))
    duplicate["notification"]["data"]["username"] = "different"
    with pytest.raises(ValueError):
        events.put(duplicate)


def test_retry_and_lease_expiry_survive_restart(receiver, signed):
    _, events = receiver
    events.put(signed[3])
    event = events.claim("mac")
    events.finish("mac", event["event_id"], event["lease"], False)
    assert events.claim("mac") is None
    with database(events.path) as db:
        db.execute("UPDATE deliveries SET until=? WHERE consumer='mac'", (time.time() - 1,))
    assert events.claim("mac")["lease"] != event["lease"]


def test_cleanup_only_registered_generated_data_and_tombstones(tmp_path):
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    try:
        run = tmp_path / "runs/production"
        run.mkdir(parents=True)
        (run / "result.json").write_text("private aggregate")
        cache = tmp_path / ".cache/embeddings"
        cache.mkdir(parents=True)
        (cache / "vector.json").write_text("vector")
        wardrobe = tmp_path / "runs/wardrobe"
        wardrobe.mkdir()
        (wardrobe / "items.json").write_text("owned")
        registry.register(run, identifiers={"username": "seller-a"})
        registry.register(cache, "cache")
        event = {"event_id": "id", "identifiers": {"username": "seller-a"}}
        assert registry.cleanup(event)["removed_artifacts"] == 2
        assert not run.exists() and not cache.exists() and wardrobe.exists()
        assert registry.cleanup(event)["status"] == "already_completed"
        with pytest.raises(DeletedAccount):
            registry.register(tmp_path / "runs/new", identifiers={"userId": "seller-a"})
    finally:
        registry.close()


def test_cleanup_preflight_rejects_symlink_without_deleting_other_files(tmp_path):
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    run = tmp_path / "runs/a"
    run.mkdir(parents=True)
    registry.register(run)
    outside = tmp_path / "private"
    outside.mkdir()
    run.rmdir()
    run.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        registry.cleanup({"event_id": "x", "identifiers": {"username": "x"}})
    assert outside.exists()
    registry.close()


@pytest.mark.parametrize(
    "relative",
    ["data/raw/wardrobe", "runs/wardrobe", ".cache/huggingface", "runs", "../outside", ".secrets"],
)
def test_protected_paths(tmp_path, relative):
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    with pytest.raises(ValueError):
        registry.register(tmp_path / relative)
    registry.close()


def test_invalidated_result_cannot_be_written_back(tmp_path, monkeypatch):
    monkeypatch.setenv("EBAY_MANAGED_ROOT", str(tmp_path))
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    registry.cleanup({"event_id": "x", "identifiers": {"username": "x"}})
    registry.close()
    with pytest.raises(DeletedAccount):
        save_report(
            {"environment": "production", "lifecycle_generation": 0},
            tmp_path / "runs/new/result.json",
        )


def test_receiver_worker_end_to_end(receiver, signed, tmp_path):
    app, events = receiver
    invoke(app, "POST", app.callback, signed[0], signature=signed[1])
    registry = Registry(tmp_path / "mac", tmp_path / "mac/.cache/ebay-lifecycle/registry.sqlite3")
    run = tmp_path / "mac/runs/api"
    run.mkdir(parents=True)
    registry.register(run)

    class Session:
        def request(self, method, url, **kwargs):
            code, body = invoke(
                app,
                method,
                url.removeprefix("https://example.com"),
                json.dumps(kwargs["json"]).encode(),
                kwargs["headers"]["Authorization"],
            )

            class Response:
                status_code = code

                def json(self):
                    return body

            return Response()

    worker = Worker("https://example.com", "a" * 40, registry, Session())
    assert worker.once()["mode"] == "status_only" and run.exists()
    assert worker.once(True)["status"] == "completed" and not run.exists()
    assert events.health()["queue_counts"] == {"done": 1, "pending": 1}
    worker.close()


def test_public_key_cache_and_dependency_failure(receiver, signed):
    class Client:
        token = "test"
        calls = 0

        def authenticate(self):
            pass

        def _json(self, *args, **kwargs):
            self.calls += 1
            return {"key": "public"}

    client = Client()
    keys = PublicKeys(client)
    assert keys("key") == keys("key") and client.calls == 1
    from travel_agent.products.retrieval.ebay.client import EbayError

    app, _ = receiver

    class Offline:
        def verify(self, *args):
            raise EbayError("failure")

    app.verifier = Offline()
    assert invoke(app, "POST", app.callback, signed[0], signature=signed[1])[0] == 503


def test_official_ebay_sdk_signature_fixture():
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "official_signature_fixture.json").read_text())
    verifier = Verifier(lambda kid: fixture["response"])
    raw = json.dumps(fixture["message"], separators=(",", ":")).encode()
    assert verifier.verify(raw, fixture["signature"]) == fixture["message"]


def test_worker_network_ack_retry_is_idempotent(receiver, signed, tmp_path):
    app, events = receiver
    events.put(signed[3])
    event = events.claim("mac")
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    assert registry.cleanup(event)["status"] == "completed"
    with database(events.path) as db:
        db.execute("UPDATE deliveries SET until=0 WHERE consumer='mac'")
    retry = events.claim("mac")
    assert registry.cleanup(retry)["status"] == "already_completed"
    events.finish("mac", retry["event_id"], retry["lease"], True)
    registry.close()


def test_worker_refuses_external_http_and_redirects(tmp_path):
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    with pytest.raises(ValueError):
        Worker("http://example.com", "a" * 40, registry)

    class Session:
        def request(self, *args, **kwargs):
            assert kwargs["allow_redirects"] is False

            class Response:
                status_code = 302

            return Response()

    worker = Worker("https://example.com", "a" * 40, registry, Session())
    with pytest.raises(RuntimeError):
        worker.once()
    worker.close()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/a",
        "https://localhost/a",
        "https://127.0.0.1/a",
        "https://example.com/internal/claim",
        "https://example.com/healthz",
    ],
)
def test_callback_requires_public_https_and_separate_routes(receiver, endpoint):
    app, events = receiver
    with pytest.raises(ValueError):
        Application(endpoint, "v" * 40, app.workers, events, app.verifier)


def test_verified_event_restart_persistence(receiver, signed):
    _, events = receiver
    events.put(signed[3])
    key = events.cipher._signing_key + events.cipher._encryption_key
    restarted = Events(events.path, base64.urlsafe_b64encode(key).decode(), ["mac", "other"])
    assert restarted.claim("mac")["identifiers"]["username"] == "seller-a"


def test_notification_id_retry_without_attempt_conflict(receiver, signed):
    _, events = receiver
    events.put(signed[3])
    duplicate = json.loads(json.dumps(signed[3]))
    duplicate["notification"]["publishAttemptCount"] = 5
    events.put(duplicate)
    assert events.health()["queue_counts"]["pending"] == 2


def test_registry_accepts_mac_system_path_aliases():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = Registry(root, root / ".cache/ebay-lifecycle/registry.sqlite3")
        path = root / "runs/test"
        path.mkdir(parents=True)
        registry.register(path)
        registry.register(path.resolve())
        assert registry.db.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 1
        registry.close()


def test_managed_export_is_removed_with_source(tmp_path, monkeypatch):
    from travel_agent.products.notifications.lifecycle import export_report

    monkeypatch.setenv("EBAY_MANAGED_ROOT", str(tmp_path))
    source = tmp_path / "runs/api/result.json"
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    registry.register(source.parent)
    registry.close()
    save_report({"environment": "production", "lifecycle_generation": 0}, source)
    target = tmp_path / "runs/handoff/result.json"
    export_report(tmp_path, source, target)
    assert target.exists()
    registry = Registry(tmp_path, tmp_path / ".cache/ebay-lifecycle/registry.sqlite3")
    registry.cleanup({"event_id": "event", "identifiers": {"userId": "test"}})
    registry.close()
    assert not source.exists() and not target.exists()


def test_service_factory_starts_without_oauth_or_model_download(tmp_path, monkeypatch):
    from travel_agent.products.notifications.service import create_app
    from travel_agent.products.retrieval.ebay.client import BrowseClient

    values = {
        "EBAY_WORKER_TOKENS_JSON": json.dumps({"mac": "a" * 40}),
        "EBAY_NOTIFICATION_DB": str(tmp_path / "queue.sqlite3"),
        "EBAY_QUEUE_KEY": Fernet.generate_key().decode(),
        "EBAY_ENVIRONMENT": "production",
        "EBAY_CLIENT_ID": "synthetic-PRD-app",
        "EBAY_CLIENT_SECRET": "synthetic-secret",
        "EBAY_NOTIFICATION_ENDPOINT": "https://example.com/ebay/account-deletion",
        "EBAY_VERIFICATION_TOKEN": "v" * 40,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        BrowseClient, "authenticate", lambda _: pytest.fail("no network at startup")
    )
    app = create_app()
    assert invoke(app, "GET", "/healthz")[0] == 200
    assert invoke(app, "GET", app.callback, query="challenge_code=local")[0] == 200
