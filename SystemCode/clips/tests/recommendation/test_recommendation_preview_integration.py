"""Stage 14 through the main-agent tool: `validate_and_recommend` -> service -> provider.

Offline only. The real provider is selected by the server environment
(`PREVIEW_PROVIDER=openai`); here it is either replaced by a fake via
`preview_provider.create_live_provider`, or the real `OpenAIImagePreviewProvider` is built by the
unchanged default factory around a fake `openai` SDK module. No test reads a `.env`, opens a
network connection or calls a paid API.

The autouse fixture in conftest.py removes the PREVIEW_* variables before each test; every test
below that depends on them sets (or clears) them explicitly in its own body. The Student
4-shaped payloads are MOCK contract fixtures (mock compatibility scores).
"""

import base64
import io
import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from travel_agent.recommendation import preview_provider
from travel_agent.recommendation.student5.models import (
    PreviewResult,
    PreviewStatus,
)
from travel_agent.recommendation.student5.preview_adapter import (
    PREVIEW_DISCLAIMER,
    PREVIEW_ENV_KEYS,
    PreviewConfigurationError,
    PreviewInputError,
    PreviewOutputError,
    PreviewProviderError,
)
from travel_agent.recommendation.tools import validate_and_recommend

ROOT = Path(__file__).resolve().parents[2]
FAKE_KEY = "sk-test-not-a-real-key-0123456789"
FAKE_MODEL = "fake-image-model"
LOGGER = "travel_agent.recommendation.preview_provider"

REQUEST = {
    "destination": "Seoul",
    "start_date": "2026-12-20",
    "end_date": "2026-12-27",
    "activities": ["sightseeing"],
    "budget": "300",
    "currency": "SGD",
    "style": ["minimalist"],
}


def _item(pid, category, price_minor):
    return {
        "product_id": pid,
        "source": "mock_contract_fixture",
        "title": pid,
        "category": category,
        "price_minor": price_minor,
        "currency": "SGD",
        "availability": True,
        "is_user_owned": False,
        "product_score": 0.6,
        "product_url": f"https://shop.example.com/{pid}",
    }


def _outfit(oid, compatibility):
    return {
        "outfit_id": oid,
        "compatibility_score": compatibility,
        "items": [
            _item(f"{oid}_top", "TOP", 5000),
            _item(f"{oid}_bottom", "BOTTOM", 5000),
            _item(f"{oid}_shoes", "FOOTWEAR", 5000),
        ],
    }


STUDENT4 = {
    "status": "ok",
    "model_version": "mock-compat-0.1",
    "source": "mock_contract_fixture",
    "outfits": [_outfit("o_a", 0.9), _outfit("o_b", 0.8), _outfit("o_c", 0.7)],
}


def _tool(payload):
    return validate_and_recommend.invoke({"payload": payload})


@pytest.fixture(scope="module")
def confirmed_outfit():
    """Recommend and confirm through the LangChain tool, as the main agent would."""
    first = _tool({"action": "recommend", "request": REQUEST, "student4": STUDENT4})
    assert first["status"] == "ok"
    confirmed = _tool(
        {
            "action": "confirm",
            "state": first["data"]["state"],
            "recommendation_set": first["data"]["recommendation_set"],
            "outfit_id": "o_a",
        }
    )
    assert confirmed["status"] == "ok"
    return confirmed["data"]["confirmed_outfit"]


def _preview(confirmed_outfit, mode="FLAT_LAY", **extra):
    return _tool(
        {"action": "preview", "confirmed_outfit": confirmed_outfit, "preview_mode": mode, **extra}
    )


def _server_env(monkeypatch, **overrides):
    """Explicit server configuration: complete `openai` settings unless overridden.

    A value of None removes that variable.
    """
    values = {
        "PREVIEW_PROVIDER": "openai",
        "PREVIEW_API_KEY": FAKE_KEY,
        "PREVIEW_MODEL": FAKE_MODEL,
        "PREVIEW_ALLOWED_IMAGE_HOSTS": None,
        **overrides,
    }
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def _clear_server_env(monkeypatch):
    """Explicitly no server preview configuration."""
    for name in PREVIEW_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


class FakeLiveProvider:
    """Stands in for OpenAIImagePreviewProvider; records the requests it receives."""

    provider_name = "openai"

    def __init__(self, failure=None):
        self.failure = failure
        self.requests = []

    def generate_preview(self, request):
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return PreviewResult(
            selected_outfit_id=request.selected_outfit_id,
            status=PreviewStatus.SUCCESS,
            requested_mode=request.preview_mode,
            delivered_mode=request.preview_mode,
            output_reference="artifacts/previews/fake.png",
            provider_name=self.provider_name,
            model_name=FAKE_MODEL,
            disclaimer=PREVIEW_DISCLAIMER,
            warnings=[],
            generation_time_seconds=0.0,
        )


@pytest.fixture
def factory(monkeypatch):
    """Replace the real-provider factory; records each server environment it is given."""
    calls = []
    provider = FakeLiveProvider()

    def fake_factory(environ):
        calls.append(dict(environ))
        return provider

    monkeypatch.setattr(preview_provider, "create_live_provider", fake_factory)
    return types.SimpleNamespace(calls=calls, provider=provider)


def _assert_fallback(result):
    assert result["status"] == "ok"
    assert result["data"]["preview_source"] == "product_card_fallback"
    preview = result["data"]["preview"]
    assert (preview["status"], preview["delivered_mode"]) == ("FALLBACK", "PRODUCT_CARDS")
    assert preview["output_reference"] is None
    assert preview["warnings"] and "product cards" in preview["warnings"][0]
    return preview


def _assert_no_secret(text):
    if not isinstance(text, str):
        text = json.dumps(text)
    assert FAKE_KEY not in text
    assert "sk-" not in text
    for path in (str(ROOT), str(Path.home())):
        assert path not in text and json.dumps(path)[1:-1] not in text


def _log_text(caplog):
    return "\n".join(r.getMessage() for r in caplog.records if r.name == LOGGER)


# Exception messages that a careless provider might produce: they must never reach the log or
# the tool result.
LEAKY_DETAILS = (
    FAKE_KEY,
    "https://img.shop.example.com/private/item.png?token=abc123",
    "C:/Users/someone/Pictures/portrait.jpg",
    "user said: my name is Alice and I live at 1 Example Road",
)
LEAKY_MESSAGE = "upstream failure: " + " | ".join(LEAKY_DETAILS)


def _assert_log_line(logs, category, exc_type):
    assert (
        f"Live preview unavailable; using product cards: category={category} type={exc_type}"
        in logs
    )


# ---------------------------------------------------------------------------
# Test environment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PREVIEW_PROVIDER parsing and server-side selection through the tool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None])
def test_unset_or_empty_preview_provider_gives_the_normal_fallback(
    confirmed_outfit, factory, monkeypatch, caplog, value
):
    _clear_server_env(monkeypatch)
    if value is not None:
        monkeypatch.setenv("PREVIEW_PROVIDER", value)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        preview = _assert_fallback(_preview(confirmed_outfit))
    assert "a preview provider is not configured" in preview["warnings"][0]
    assert factory.calls == []
    assert _log_text(caplog) == ""


@pytest.mark.parametrize("mode", ["FLAT_LAY"])
def test_preview_provider_openai_reaches_the_live_provider_from_the_tool(
    confirmed_outfit, factory, monkeypatch, mode
):
    _server_env(monkeypatch)
    result = _preview(confirmed_outfit, mode)
    assert result["status"] == "ok"
    assert result["data"]["preview_source"] == "openai"
    preview = result["data"]["preview"]
    assert (preview["status"], preview["delivered_mode"]) == ("SUCCESS", mode)
    assert preview["output_reference"] == "artifacts/previews/fake.png"
    # The factory got the server environment; the provider got this one confirmed outfit.
    assert len(factory.calls) == 1
    assert factory.calls[0]["PREVIEW_API_KEY"] == FAKE_KEY
    [request] = factory.provider.requests
    assert request.selected_outfit_id == "o_a"
    assert request.user_portrait is None and request.portrait_consent is False
    _assert_no_secret(result)


# ---------------------------------------------------------------------------
# Known failures fall back; programming errors propagate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure, category, warning",
    [
        (PreviewConfigurationError(LEAKY_MESSAGE), "configuration", "not configured correctly"),
        (PreviewInputError(LEAKY_MESSAGE, reason="download"), "input", "input images"),
        (PreviewOutputError(LEAKY_MESSAGE), "output", "output could not"),
        (PreviewProviderError(LEAKY_MESSAGE), "provider", "request failed"),
    ],
)
def test_known_provider_failures_fall_back_and_log_only_category_and_type(
    confirmed_outfit, factory, monkeypatch, caplog, failure, category, warning
):
    factory.provider.failure = failure
    _server_env(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = _preview(confirmed_outfit, "FLAT_LAY")
    preview = _assert_fallback(result)
    assert warning in preview["warnings"][0]
    assert len(factory.provider.requests) == 1
    logs = _log_text(caplog)
    _assert_log_line(logs, category, type(failure).__name__)
    result_text = json.dumps(result)
    for text in (logs, result_text):
        assert str(failure) not in text
        assert "upstream failure" not in text
        for detail in LEAKY_DETAILS:
            assert detail not in text and json.dumps(detail)[1:-1] not in text
    _assert_no_secret(result)
    _assert_no_secret(logs)


# ---------------------------------------------------------------------------
# Portrait consent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {
            "preview_mode": "PORTRAIT",
            "user_portrait_ref": "portrait_test",
            "portrait_consent": False,
        },
        {"preview_mode": "FLAT_LAY", "user_portrait_ref": "portrait_test"},
    ],
)
def test_portrait_without_consent_never_reaches_a_provider(
    confirmed_outfit, factory, monkeypatch, tmp_path, extra
):
    _server_env(monkeypatch)
    portrait_store = tmp_path / "portraits"
    portrait_store.mkdir()
    (portrait_store / "portrait_test.png").write_bytes(b"test portrait placeholder")
    monkeypatch.setenv("PORTRAIT_UPLOAD_ROOT", str(portrait_store))
    extra = dict(extra)
    mode = extra.pop("preview_mode")
    result = _preview(confirmed_outfit, mode, **extra)
    assert result["status"] == "needs_input"
    assert factory.calls == [] and factory.provider.requests == []


def test_consented_portrait_is_passed_to_the_live_provider(
    confirmed_outfit, factory, monkeypatch, tmp_path
):
    # Tool -> service -> orchestrator -> fake live provider, with an uploaded, consented portrait.
    _server_env(monkeypatch)
    portrait_store = tmp_path / "portraits"
    portrait_store.mkdir()
    portrait = portrait_store / "portrait_test.png"
    portrait.write_bytes(b"test portrait placeholder")
    monkeypatch.setenv("PORTRAIT_UPLOAD_ROOT", str(portrait_store))
    result = _preview(
        confirmed_outfit, "PORTRAIT", user_portrait_ref="portrait_test", portrait_consent=True
    )
    assert result["status"] == "ok"
    assert result["data"]["preview_source"] == "openai"
    preview = result["data"]["preview"]
    assert (preview["status"], preview["delivered_mode"]) == ("SUCCESS", "PORTRAIT")
    [request] = factory.provider.requests
    assert request.selected_outfit_id == "o_a"
    assert request.portrait_consent is True
    assert Path(request.user_portrait).resolve() == portrait.resolve()


# ---------------------------------------------------------------------------
# The payload cannot select or configure the real provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"preview_provider": "openai"},
        {"PREVIEW_API_KEY": FAKE_KEY},
        {"allowed_hosts": ["evil.example.com"]},
        {"output_dir": "C:/Windows"},
    ],
)
def test_payload_cannot_select_or_configure_the_real_provider(
    confirmed_outfit, factory, monkeypatch, extra
):
    _server_env(monkeypatch)
    result = _preview(confirmed_outfit, **extra)
    assert result["status"] == "needs_input"
    assert factory.calls == []
    _assert_no_secret(result)


# ---------------------------------------------------------------------------
# Real OpenAIImagePreviewProvider behind the tool, with a fake `openai` SDK
# ---------------------------------------------------------------------------


def _png_b64():
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (8, 8), (200, 180, 160)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _fake_openai_sdk(b64, *, init_error=None, generate_error=None):
    sdk = types.ModuleType("openai")
    sdk.created = []
    sdk.calls = []

    class OpenAIError(Exception):
        pass

    sdk.OpenAIError = OpenAIError

    class Images:
        def _respond(self, method, kwargs):
            sdk.calls.append((method, kwargs))
            if generate_error is not None:
                raise OpenAIError(generate_error)
            return types.SimpleNamespace(data=[types.SimpleNamespace(b64_json=b64)])

        def generate(self, **kwargs):
            return self._respond("generate", kwargs)

        def edit(self, **kwargs):
            return self._respond("edit", kwargs)

    class OpenAI:
        def __init__(self, api_key):
            if init_error is not None:
                raise OpenAIError(init_error)
            sdk.created.append(api_key)
            self.images = Images()

    sdk.OpenAI = OpenAI
    return sdk


def _files_under(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file())


def test_default_factory_builds_the_real_provider_behind_the_tool(
    confirmed_outfit, monkeypatch, tmp_path
):
    """validate_and_recommend.invoke -> service._preview -> server_preview_provider ->
    create_live_provider -> OpenAIImagePreviewProvider.from_env -> openai.OpenAI(...).images."""
    b64 = _png_b64()  # skips without Pillow (the preview extra)
    sdk = _fake_openai_sdk(b64)
    monkeypatch.setitem(sys.modules, "openai", sdk)
    monkeypatch.chdir(tmp_path)  # default output directory artifacts/previews is relative
    _server_env(monkeypatch)

    result = _preview(confirmed_outfit, "FLAT_LAY")

    assert result["status"] == "ok", result
    assert result["data"]["preview_source"] == "openai"
    preview = result["data"]["preview"]
    assert (preview["status"], preview["provider_name"], preview["model_name"]) == (
        "SUCCESS",
        "openai",
        FAKE_MODEL,
    )
    assert sdk.created == [FAKE_KEY]  # key from the server environment only
    [(method, kwargs)] = sdk.calls
    assert method == "generate"  # the fixture products have no image to send
    assert kwargs["model"] == FAKE_MODEL and kwargs["n"] == 1
    output = Path(preview["output_reference"])
    assert not output.is_absolute() and output.parts[:2] == ("artifacts", "previews")
    assert (tmp_path / output).is_file()
    assert b64 not in json.dumps(result)
    _assert_no_secret(result)


# ---------------------------------------------------------------------------
# Import safety with the real provider selected
# ---------------------------------------------------------------------------

IMPORT_PROBE = """
import builtins, io, os, socket, sys
sys.path.insert(0, sys.argv[1])
def _blocked(*a, **k):
    raise RuntimeError("network access attempted")
socket.socket.connect = _blocked
socket.create_connection = _blocked
opened = []
for owner, name in ((builtins, "open"), (io, "open")):
    real = getattr(owner, name)
    def recording(file, *a, _real=real, **k):
        opened.append(str(file))
        return _real(file, *a, **k)
    setattr(owner, name, recording)
read_keys = []
real_getitem = os._Environ.__getitem__
def recording_getitem(self, key, _real=real_getitem):
    read_keys.append(key)
    return _real(self, key)
os._Environ.__getitem__ = recording_getitem
heavy = ("openai", "PIL")
before = {m for m in heavy if m in sys.modules}
import travel_agent.recommendation.service
import travel_agent.recommendation.tools
after = {m for m in heavy if m in sys.modules}
assert not {"openai", "PIL"} & (after - before), sorted(after - before)
assert not [p for p in opened if os.path.basename(p).startswith(".env")], opened
assert not [k for k in read_keys if k.startswith("PREVIEW_")], read_keys
assert sorted(os.listdir(".")) == [], os.listdir(".")
print("ok")
"""


def test_import_reads_no_preview_config_and_creates_nothing_even_when_openai_is_selected(
    tmp_path,
):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PREVIEW_", "PYTHON"))}
    env.update(
        {"PREVIEW_PROVIDER": "openai", "PREVIEW_API_KEY": FAKE_KEY, "PREVIEW_MODEL": FAKE_MODEL}
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", IMPORT_PROBE, str(ROOT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
