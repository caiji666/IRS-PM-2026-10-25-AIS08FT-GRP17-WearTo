"""Proposal Stages 1-3; deterministic orchestration with explicit partial output."""

from typing import Any

from pydantic import ValidationError

from travel_agent.schemas import (
    ConversationState,
    EnrichedTripContext,
    ModuleResult,
    TripShoppingRequest,
    TripShoppingRequestPatch,
)
from travel_agent.understanding.extraction import DIALOGUE_SCHEMA, QwenExtractor
from travel_agent.understanding.knowledge import reason_activities, reason_places
from travel_agent.understanding.rules import reason_weather
from travel_agent.understanding.season import derive_season
from travel_agent.understanding.state import apply_patch, confirm_outfit, update_state
from travel_agent.understanding.weather import FixtureWeather, OpenMeteoWeather, UnavailableWeather

REQUIRED_FIELDS = ("destination", "start_date", "end_date", "activities", "budget", "currency")


def required_questions(request):
    return [
        {
            "field": field,
            "question": f"Please provide {field}.",
            "reason": "Needed for the complete travel-shopping workflow",
        }
        for field in REQUIRED_FIELDS
        if not getattr(request, field)
    ]


def expand_context(request, weather, clothes, activities, places, season_context=None):
    # Keep activity requirements separate structurally; text is for retrieval,
    # never a substitute for hard-constraint checks or outfit feasibility.
    words = [request.destination or ""]
    if season_context is not None:
        words.extend(season_context.seasons)
    if weather.weather_class:
        words.append(weather.weather_class.lower())
    for values in clothes.model_dump().values():
        words.extend(values)
    for activity in request.activities:
        words.extend([activity, *activities.get(activity, [])])
    words.extend(request.style)
    if places:
        words.extend(r["req_id"].replace("_", " ") for r in places["requirements"])
    return " ".join(" ".join(words).split())


def run(
    payload: dict, *, extractor=None, weather_provider=None, activity_graph=None
) -> ModuleResult:
    """Input: request OR patch OR message, optional serialized state and settings.

    State is returned even on missing information or invalid edits. The caller
    persists it per session; the service never keeps shared global chat state.
    """
    state = ConversationState()
    extraction = {}
    try:
        state = ConversationState.model_validate(payload.get("state", {}))
        state_loaded = state.model_copy(deep=True)
        if type(payload.get("accept_extraction", False)) is not bool:
            raise ValueError("accept_extraction must be boolean")
        mode = payload.get("intent")
        if mode == "CONFIRM_OUTFIT":
            state = confirm_outfit(state, payload.get("outfit_id"), payload.get("display_position"))
            return ModuleResult(status="ok", data={"state": state.model_dump(mode="json")})
        if mode == "REQUEST_EXPLANATION":
            return ModuleResult(
                status="needs_input",
                data={"state": state.model_dump(mode="json")},
                reason="The recommendation module supplies outfit explanations",
            )
        inputs = [key for key in ("request", "patch", "message") if key in payload]
        if len(inputs) != 1:
            raise ValueError("Supply exactly one of request, patch or message")
        extraction: dict[str, Any] = {"provider": "structured_input"}
        if "request" in payload:
            if state.request_version:
                raise ValueError(
                    "Use a patch for an existing session instead of replacing the request"
                )
            request = TripShoppingRequest.model_validate(payload["request"])
        elif "patch" in payload:
            patch = TripShoppingRequestPatch.model_validate(payload["patch"])
            request = apply_patch(state.current_request, patch)
        else:
            message = payload["message"]
            if not isinstance(message, str) or not message.strip():
                raise ValueError("Message must be nonempty text")
            extractor = extractor or QwenExtractor()
            previous = state.current_request if state.request_version else None
            extracted, extraction = extractor.extract(message, previous)
            if previous:
                action = DIALOGUE_SCHEMA.validate_python(extracted)
                if action.intent == "CONFIRM_OUTFIT":
                    if not payload.get("accept_extraction", False):
                        return ModuleResult(
                            status="needs_input",
                            reason="Review outfit selection",
                            data={
                                "state": state.model_dump(mode="json"),
                                "extraction": extraction,
                                "proposed_action": action.model_dump(mode="json"),
                                "questions": [
                                    {
                                        "field": "extraction_review",
                                        "question": (
                                            "Confirm the extracted outfit ID/position "
                                            "before applying it."
                                        ),
                                        "reason": "Selection requires caller review",
                                    }
                                ],
                            },
                        )
                    state = confirm_outfit(state, action.outfit_id, action.display_position)
                    return ModuleResult(
                        status="ok",
                        data={"state": state.model_dump(mode="json"), "extraction": extraction},
                    )
                if action.intent == "REQUEST_EXPLANATION":
                    return ModuleResult(
                        status="needs_input",
                        data={"state": state.model_dump(mode="json"), "extraction": extraction},
                        reason="The recommendation module supplies outfit explanations",
                    )
            request = (
                apply_patch(
                    state.current_request, TripShoppingRequestPatch.model_validate(extracted)
                )
                if previous
                else TripShoppingRequest.model_validate(extracted)
            )
        if "message" in payload and not payload.get("accept_extraction", False):
            return ModuleResult(
                status="needs_input",
                reason="Human extraction review required",
                data={
                    "state": state.model_dump(mode="json"),
                    "proposed_request": request.model_dump(mode="json"),
                    "proposed_action": extracted if previous else None,
                    "extraction": extraction,
                    "questions": [
                        {
                            "field": "extraction_review",
                            "question": "Check the extracted facts against your message. Submit "
                            "the corrected structured request/patch to continue.",
                            "reason": "Qwen3-0.6B semantic extraction is not validated",
                        }
                    ],
                },
            )
        state = update_state(state, request)
        # An empty first request still creates a collecting session.
        if not state.request_version:
            state.request_version = 1
        questions = required_questions(request)
        season_context = derive_season(
            request, confirmed_location=payload.get("confirmed_location")
        )
        base = {
            "season": season_context.season,
            "season_context": season_context.model_dump(mode="json"),
            "trip_request": request.model_dump(mode="json"),
            "state": state.model_dump(mode="json"),
            "questions": questions,
            "extraction": extraction,
            "contract_version": "proposal-appendix-a-v1",
        }
        if questions:
            return ModuleResult(
                status="needs_input",
                data=base,
                reason="Complete or clarify the mandatory request fields",
            )
        allow_draft = payload.get("allow_draft", False)
        if type(allow_draft) is not bool:
            raise ValueError("allow_draft must be boolean")
        provider = weather_provider
        if provider is None:
            if "weather_fixture" in payload:
                provider = FixtureWeather(payload["weather_fixture"])
            elif payload.get("weather_provider") == "open_meteo":
                provider = OpenMeteoWeather(location=payload.get("confirmed_location"))
            else:
                provider = UnavailableWeather()
        weather, clothing, trace = reason_weather(provider.fetch(request))
        season_context = derive_season(request, weather, payload.get("confirmed_location"))
        base["season"] = season_context.season
        base["season_context"] = season_context.model_dump(mode="json")
        if activity_graph is None and payload.get("kg_backend") == "neo4j":
            from travel_agent.understanding.neo4j_backend import Neo4jKnowledgeGraph

            try:
                activity_graph = Neo4jKnowledgeGraph.from_environment()
            except (ImportError, ValueError):
                raise
            except Exception as exc:
                raise RuntimeError(f"Neo4j connection unavailable: {type(exc).__name__}") from None
        try:
            activity_styles, activity_results, activity_questions = reason_activities(
                request, allow_draft, payload.get("kg_max_depth", 4), activity_graph
            )
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            if payload.get("kg_backend") == "neo4j":
                raise RuntimeError(f"Neo4j query unavailable: {type(exc).__name__}") from None
            raise
        questions.extend(activity_questions)
        try:
            places = reason_places(request, allow_draft)
        except ValueError as exc:
            places = {"status": "unsupported_place", "reason": str(exc), "requirements": []}
            questions.append(
                {
                    "field": "places",
                    "question": "Please clarify the intended POI",
                    "reason": str(exc),
                }
            )
        if weather.status == "needs_location":
            questions.append(
                {
                    "field": "confirmed_location",
                    "question": "Which city/location do you mean?",
                    "reason": weather.reason,
                    "candidates": weather.location_candidates,
                }
            )
        limitations = []
        if weather.status != "ok":
            limitations.append(weather.reason or "Weather is unknown")
        elif weather.data_type != "forecast":
            limitations.append(f"Weather is {weather.data_type}; it is not a trip-date forecast")
        incomplete = [a for a, r in activity_results.items() if r["status"] != "ok"]
        if incomplete:
            limitations.append("No complete KG mapping for: " + ", ".join(incomplete))
        if allow_draft:
            limitations.append("Draft KG knowledge explicitly enabled; human review pending")
        if trace:
            limitations.append("Weather rules are prototype heuristics pending human review")
        state.session_status = "COLLECTING_REQUIREMENTS" if questions else "READY"
        # Preserve confirmation history on a no-op; no recommendations are created here.
        if request == state_loaded.current_request and state_loaded.session_status in {
            "CONFIRMED",
            "AWAITING_CONFIRMATION",
        }:
            state.session_status = state_loaded.session_status
        enriched = EnrichedTripContext(trip_request=request, weather_context=weather)
        base.update(
            {
                "state": state.model_dump(mode="json"),
                "enriched_trip_context": enriched.model_dump(mode="json"),
                "weather_clothing_requirements": clothing.model_dump(),
                "activity_style_requirements": activity_styles,
                "activity_reasoning": activity_results,
                "weather_rule_trace": trace,
                "place_context": places,
                "expanded_fashion_context": expand_context(
                    request, weather, clothing, activity_styles, places, season_context
                ),
                "hard_constraints": {
                    "budget": request.model_dump(mode="json")["budget"],
                    "currency": request.currency,
                    "budget_minor": request.budget_minor_units(),
                    "excluded_items": request.excluded_items,
                    "owned_item_ids": request.owned_item_ids,
                },
                "questions": questions,
                "limitations": limitations,
                "context_status": "partial" if limitations or questions else "complete",
                "ready_for_downstream": not questions,
                "feasibility_status": "unverified",
            }
        )
        return ModuleResult(
            status="needs_input" if questions else "ok",
            data=base,
            reason="Clarification required" if questions else None,
        )
    except (ValidationError, ValueError) as exc:
        if getattr(exc, "raw_output", None):
            extraction = {
                "provider": "qwen",
                "raw_output": exc.raw_output,
                "human_review_required": True,
            }
        errors = (
            [
                {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                for e in exc.errors()
            ]
            if isinstance(exc, ValidationError)
            else [{"field": "input", "message": str(exc)}]
        )
        return ModuleResult(
            status="needs_input",
            reason="Input/extraction needs correction",
            data={
                "state": state.model_dump(mode="json"),
                "validation_errors": errors,
                "extraction": extraction,
                "questions": [
                    {
                        "field": "input",
                        "question": "Please correct or clarify the request",
                        "reason": str(exc),
                    }
                ],
            },
        )
    except (OSError, ImportError, RuntimeError) as exc:
        return ModuleResult(
            status="error",
            reason=f"Provider unavailable: {type(exc).__name__}",
            data={"state": state.model_dump(mode="json"), "detail": str(exc)},
        )
