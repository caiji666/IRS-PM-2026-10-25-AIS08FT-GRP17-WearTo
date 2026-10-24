"""Main-agent integration for Stage 14: `build_agent(model).invoke(...)` -> outfit preview.

Fully offline. The test starts from the shared production agent, `travel_agent.agent.build_agent`,
with a scripted tool-calling chat model defined here (test-only, never part of the package). The
server environment selects the real provider (`PREVIEW_PROVIDER=openai`), but
`preview_provider.create_live_provider` is replaced by a fake provider, sockets are blocked, and
no API is called.

What this proves: the agent's technical call chain works,

    build_agent(fake_model).invoke -> validate_and_recommend -> recommendation.service
    -> server-configured provider -> PreviewResult -> tool result -> final assistant reply.

What it cannot prove: that a real LLM always judges the user's intent correctly. The scripted
model decides by script. "Generate a preview only when the user explicitly asks" remains a
prompt and tool-contract rule, not a code-level guarantee.

The Student 4-shaped payload is a MOCK contract fixture (mock compatibility scores).
"""

import json
import socket
import types
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from travel_agent.agent import build_agent
from travel_agent.recommendation import preview_provider
from travel_agent.recommendation.service import run as student5_run
from travel_agent.recommendation.student5.models import PreviewResult, PreviewStatus
from travel_agent.recommendation.student5.preview_adapter import (
    PREVIEW_DISCLAIMER,
    PREVIEW_ENV_KEYS,
)

FAKE_KEY = "sk-test-not-a-real-key-0123456789"
FAKE_MODEL = "fake-image-model"
TOOL = "validate_and_recommend"
FORBIDDEN_ARG_KEYS = {
    "api_key",
    "PREVIEW_API_KEY",
    "model",
    "PREVIEW_MODEL",
    "allowed_hosts",
    "PREVIEW_ALLOWED_IMAGE_HOSTS",
    "output_dir",
    "output_path",
    "env_file",
    ".env",
    "PREVIEW_PROVIDER",
}

REQUEST = {
    "destination": "Seoul",
    "start_date": "2026-12-20",
    "end_date": "2026-12-27",
    "activities": ["sightseeing"],
    "budget": "300",
    "currency": "SGD",
    "style": ["minimalist"],
}


def _item(pid, category):
    return {
        "product_id": pid,
        "source": "mock_contract_fixture",
        "title": pid,
        "category": category,
        "price_minor": 5000,
        "currency": "SGD",
        "availability": True,
        "is_user_owned": False,
        "product_score": 0.6,
        "product_url": f"https://shop.example.com/{pid}",
    }


STUDENT4 = {
    "status": "ok",
    "model_version": "mock-compat-0.1",
    "source": "mock_contract_fixture",
    "outfits": [
        {
            "outfit_id": oid,
            "compatibility_score": score,
            "items": [
                _item(f"{oid}_top", "TOP"),
                _item(f"{oid}_bottom", "BOTTOM"),
                _item(f"{oid}_shoes", "FOOTWEAR"),
            ],
        }
        for oid, score in (("o_a", 0.9), ("o_b", 0.8), ("o_c", 0.7))
    ],
}


# ---------------------------------------------------------------------------
# Test-only scripted tool-calling chat model
# ---------------------------------------------------------------------------


class ScriptedToolCallingModel(BaseChatModel):
    """Minimal chat model for create_agent: each call runs the next scripted step.

    A step is a function `messages -> AIMessage`. `bind_tools` records the tool names offered
    by the agent and returns the model itself. The message list of every call is kept in
    `script.calls`, so a test can check how often the agent called the model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: Any  # SimpleNamespace(steps=[...], calls=[...], bound_tools=[...])

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.script.bound_tools.append(sorted(getattr(t, "name", str(t)) for t in tools))
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        position = len(self.script.calls)
        self.script.calls.append(list(messages))
        if position >= len(self.script.steps):
            raise AssertionError(f"model called {position + 1} times; script has no more steps")
        return ChatResult(
            generations=[ChatGeneration(message=self.script.steps[position](messages))]
        )


def _scripted_model(*steps):
    return ScriptedToolCallingModel(
        script=types.SimpleNamespace(steps=list(steps), calls=[], bound_tools=[])
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeLiveProvider:
    """Stands in for OpenAIImagePreviewProvider (no SDK, no network, no cost)."""

    provider_name = "openai"

    def __init__(self):
        self.requests = []

    def generate_preview(self, request):
        self.requests.append(request)
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
def offline_server(monkeypatch):
    """Server environment selects `openai`; the provider factory is faked; sockets blocked."""
    for name in PREVIEW_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PREVIEW_PROVIDER", "openai")
    monkeypatch.setenv("PREVIEW_API_KEY", FAKE_KEY)
    monkeypatch.setenv("PREVIEW_MODEL", FAKE_MODEL)

    network = []

    def blocked(*args, **kwargs):
        network.append(args)
        raise OSError("network access is blocked in this test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    provider = FakeLiveProvider()
    factory_calls = []

    def fake_factory(environ):
        factory_calls.append(environ.get("PREVIEW_PROVIDER"))
        return provider

    monkeypatch.setattr(preview_provider, "create_live_provider", fake_factory)
    return types.SimpleNamespace(provider=provider, factory_calls=factory_calls, network=network)


@pytest.fixture(scope="module")
def confirmed_history():
    """A conversation in which the user confirmed outfit o_a through the existing
    recommend -> confirm flow. The confirm tool result carries the ConfirmedOutfit."""
    first = student5_run({"action": "recommend", "request": REQUEST, "student4": STUDENT4})
    assert first.status == "ok"
    confirm_args = {
        "action": "confirm",
        "state": first.data["state"],
        "recommendation_set": first.data["recommendation_set"],
        "outfit_id": "o_a",
    }
    confirmed = student5_run(confirm_args)
    assert confirmed.status == "ok"
    return [
        HumanMessage("I'm going to Seoul in December, please recommend outfits."),
        AIMessage(
            "",
            tool_calls=[
                {
                    "id": "call_confirm_1",
                    "name": TOOL,
                    "args": {"payload": confirm_args},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            json.dumps(confirmed.model_dump(mode="json")),
            tool_call_id="call_confirm_1",
            name=TOOL,
        ),
        AIMessage("You confirmed outfit o_a."),
    ]


def _confirmed_outfit_from(messages):
    """What a model reads from the conversation: the latest confirm tool result."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == TOOL:
            data = json.loads(message.content).get("data", {})
            if "confirmed_outfit" in data:
                return data["confirmed_outfit"]
    raise AssertionError("no confirmed outfit in the conversation")


def _new_tool_messages(result, history_length):
    return [
        m
        for m in result["messages"][history_length:]
        if isinstance(m, ToolMessage) and m.name == TOOL
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_generates_one_flat_lay_preview_on_explicit_request(
    offline_server, confirmed_history
):
    emitted = {}

    def request_preview(messages):
        assert isinstance(messages[-1], HumanMessage)
        args = {
            "payload": {
                "action": "preview",
                "confirmed_outfit": _confirmed_outfit_from(messages),
                "preview_mode": "FLAT_LAY",
            }
        }
        emitted["args"] = args
        return AIMessage(
            "",
            tool_calls=[{"id": "call_preview_1", "name": TOOL, "args": args, "type": "tool_call"}],
        )

    def answer(messages):
        tool_result = json.loads(messages[-1].content)
        emitted["tool_result_seen_by_model"] = tool_result
        source = tool_result["data"]["preview_source"]
        reference = tool_result["data"]["preview"]["output_reference"]
        return AIMessage(f"Here is your FLAT_LAY preview ({source}): {reference}")

    model = _scripted_model(request_preview, answer)
    agent = build_agent(model)
    history = list(confirmed_history)
    user_request = HumanMessage("为我刚刚确认的这一套生成 FLAT_LAY 预览。")

    result = agent.invoke({"messages": [*history, user_request]})

    # The agent offered the shared tools, and the model was called exactly twice.
    assert TOOL in model.script.bound_tools[0]
    assert len(model.script.calls) == 2

    # Tool call emitted by the model: preview only, no provider configuration.
    payload = emitted["args"]["payload"]
    assert payload["action"] == "preview" and payload["preview_mode"] == "FLAT_LAY"
    assert payload["confirmed_outfit"]["selected_outfit_id"] == "o_a"
    assert not FORBIDDEN_ARG_KEYS & set(payload)
    assert payload.get("preview_provider") is None

    # Exactly one validate_and_recommend call, reaching the server-configured provider once.
    [tool_message] = _new_tool_messages(result, len(history) + 1)
    assert offline_server.factory_calls == ["openai"]
    [provider_request] = offline_server.provider.requests
    assert provider_request.selected_outfit_id == "o_a"
    assert provider_request.portrait_consent is False and provider_request.user_portrait is None

    tool_result = json.loads(tool_message.content)
    assert tool_result["status"] == "ok"
    assert tool_result["data"]["preview_source"] == "openai"
    preview = tool_result["data"]["preview"]
    assert preview["status"] == PreviewStatus.SUCCESS.value
    assert preview["selected_outfit_id"] == "o_a"
    assert (preview["requested_mode"], preview["delivered_mode"]) == ("FLAT_LAY", "FLAT_LAY")
    assert emitted["tool_result_seen_by_model"] == tool_result

    # Final assistant reply was produced from the tool result.
    final = result["messages"][-1]
    assert isinstance(final, AIMessage) and not final.tool_calls
    assert "FLAT_LAY preview (openai)" in final.content

    # No key anywhere, no network access.
    for text in (json.dumps(emitted["args"]), tool_message.content, final.content):
        assert FAKE_KEY not in text
    assert offline_server.network == []


def test_agent_does_not_call_the_tool_when_no_preview_is_requested(
    offline_server, confirmed_history
):
    model = _scripted_model(lambda messages: AIMessage("Enjoy your trip to Seoul!"))
    agent = build_agent(model)
    history = list(confirmed_history)

    result = agent.invoke({"messages": [*history, HumanMessage("Thanks, that's all for now.")]})

    assert len(model.script.calls) == 1
    assert _new_tool_messages(result, len(history) + 1) == []
    assert offline_server.factory_calls == [] and offline_server.provider.requests == []
    assert result["messages"][-1].content == "Enjoy your trip to Seoul!"
    assert offline_server.network == []
