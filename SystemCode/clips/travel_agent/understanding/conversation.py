"""Stateful local dialogue over the deterministic understanding service.

The controller owns state, questions and approval. Qwen proposes natural wording;
invalid wording falls back to the controller's factual reference reply.
"""

import json
import re
from copy import deepcopy

from travel_agent.schemas import ConversationState, TripShoppingRequest, TripShoppingRequestPatch
from travel_agent.understanding.dates import normalize_dates
from travel_agent.understanding.extraction import QwenExtractor
from travel_agent.understanding.service import required_questions, run
from travel_agent.understanding.state import apply_patch
from travel_agent.understanding.suggestions import optional_suggestions

QUESTIONS = {
    "destination": "Where would you like to travel?",
    "start_date": "What is your departure date? Please include the year.",
    "end_date": "What is your return date? Please include the year.",
    "budget": "What is your shopping budget?",
    "currency": "Which currency is your budget in?",
}
CONFIRM = re.compile(
    r"\s*(?:/confirm|yes|yes[, ]+(?:please|confirm|that is correct)|confirm|confirmed|"
    r"that is correct|looks correct|correct|确认|是的|没错|确认无误)\s*[.!。！]?\s*",
    re.I,
)
GREETING = re.compile(r"\s*(?:hi|hello|hey|你好|您好)\s*[!.。！]?\s*", re.I)
DECLINE = re.compile(
    r"\s*(?:no(?:[, ]+thanks)?|no thank you|nothing else|that(?:'s| is) all|"
    r"no more changes|不用了|不需要|没有了|不用|不了)\s*[!.。！]?\s*",
    re.I,
)


def facts_text(request):
    parts = []
    for key, label in (
        ("destination", "a trip to"),
        ("start_date", "departing on"),
        ("end_date", "returning on"),
    ):
        if request.get(key):
            parts.append(f"{label} {request[key]}")
    if request.get("budget") is not None:
        parts.append(
            f"a shopping budget of {request['budget']} "
            f"{request.get('currency') or '(currency needed)'}"
        )
    elif request.get("currency"):
        parts.append(f"currency: {request['currency']}")
    for key, label in (
        ("activities", "activities"),
        ("style", "style"),
        ("excluded_items", "exclude"),
    ):
        if request.get(key):
            parts.append(f"{label}: {', '.join(request[key])}")
    if request.get("dress_codes"):
        parts.append(
            "dress codes: " + ", ".join(f"{a}: {c}" for a, c in request["dress_codes"].items())
        )
    return ", ".join(parts)


class QwenResponder:
    def __init__(self, extractor):
        self.extractor = extractor

    def respond(self, reference, question):
        raw = self.extractor.generate_text(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the draft as a short friendly travel assistant message. "
                        "Keep all facts and copy its final question exactly. Do not answer "
                        "the question. Do not add facts. Output only the rewritten message."
                    ),
                },
                {
                    "role": "user",
                    "content": "Draft: I have recorded that information. "
                    "What is your departure date? Please include the year.",
                },
                {
                    "role": "assistant",
                    "content": "Thanks, I have noted that. What is your departure date? "
                    "Please include the year.",
                },
                {"role": "user", "content": "Draft: " + reference},
            ],
            max_new_tokens=240,
        )
        # These lexical checks are conservative prototype guards, not semantic proof.
        valid = bool(raw.strip()) and not any(token in raw for token in ("```", "{", "<think>"))
        valid = valid and (not question or raw.rstrip().endswith(question))
        valid = valid and set(re.findall(r"\d+(?:\.\d+)?", raw)) == set(
            re.findall(r"\d+(?:\.\d+)?", reference)
        )
        place = re.search(r"a trip to ([^,;.]+)", reference, re.I)
        if place:
            valid = valid and place.group(1).casefold() in raw.casefold()
        known_words = {word.casefold() for word in re.findall(r"[A-Za-z]+", reference)}
        neutral_openers = {"thanks", "thank", "great", "got", "sure", "okay", "noted", "understood"}
        valid = valid and all(
            word.casefold() in known_words | neutral_openers
            for word in re.findall(r"\b[A-Z][a-z]+\b", raw)
        )
        valid = valid and raw.count("?") == reference.count("?")
        forbidden = ("booked", "purchased", "forecast", "sunny", "rainy", "snowy", "temperature")
        valid = valid and all(
            word not in raw.casefold() or word in reference.casefold() for word in forbidden
        )
        return (raw.strip() if valid else reference), {
            "provider": "qwen",
            "raw_reply": raw,
            "used_fallback": not valid,
        }


class Conversation:
    """One caller-owned conversation; serialize after every turn, including errors."""

    def __init__(self, *, extractor=None, responder=None, saved=None, settings=None):
        saved = saved or {}
        # Accept both a chat checkpoint and an earlier understanding/debug state.
        raw_state = saved.get("state", saved)
        self.state = ConversationState.model_validate(raw_state).model_dump(mode="json")
        self.extractor = extractor or QwenExtractor()
        self.responder = responder or QwenResponder(self.extractor)
        self.settings = dict(settings or {})
        self.awaiting_field = saved.get("awaiting_field") if "state" in saved else None
        self.last_result = saved.get("last_result") if "state" in saved else None
        self.finished = bool(saved.get("finished", False)) if "state" in saved else False
        self.optional_decisions = (
            deepcopy(saved.get("optional_decisions", {})) if "state" in saved else {}
        )
        self.offered_options = deepcopy(saved.get("offered_options")) if "state" in saved else None
        self.pending_date_source = saved.get("pending_date_source")
        self.date_question = saved.get("date_question")

    def checkpoint(self):
        return deepcopy(
            {
                "state": self.state,
                "awaiting_field": self.awaiting_field,
                "last_result": self.last_result,
                "finished": self.finished,
                "optional_decisions": self.optional_decisions,
                "offered_options": self.offered_options,
                "pending_date_source": self.pending_date_source,
                "date_question": self.date_question,
            }
        )

    def _scope(self, field, request):
        scope = {key: request[key] for key in ("destination", "places")}
        if field == "style":
            scope.update({key: request[key] for key in ("activities", "dress_codes")})
        scope["allow_draft"] = self.settings.get("allow_draft", False)
        return json.dumps(scope, sort_keys=True, ensure_ascii=False)

    def _next_optional(self, request, suggestions=None):
        suggestions = suggestions or optional_suggestions(
            TripShoppingRequest.model_validate(request),
            allow_draft=self.settings.get("allow_draft", False),
        )
        for field in ("activities", "style"):
            scope = self._scope(field, request)
            if request[field] or self.optional_decisions.get(field, {}).get("scope") == scope:
                continue
            options = suggestions[field]
            if options:
                return {"field": field, "scope": scope, "options": options}
        return None

    def _optional_turn(self, message):
        """Resolve only explicit choices from the exact last-displayed menu."""
        offered = self.offered_options
        if not offered:
            return None
        working = self.state.get("pending_request") or self.state["current_request"]
        field = offered["field"]
        if offered["scope"] != self._scope(field, working):
            self.offered_options = None
            return self._reply({"status": "needs_input", "data": {}})
        skip_all = bool(
            re.fullmatch(r"skip (?:all|both)(?: optional (?:fields|details))?", message, re.I)
        )
        skip = (
            skip_all
            or bool(DECLINE.fullmatch(message))
            or bool(
                re.fullmatch(
                    r"(?:skip|none|none of these|no preferences?|no activities|都不选|跳过|不选)",
                    message,
                    re.I,
                )
            )
        )
        if skip:
            for skipped in ("activities", "style") if skip_all else (field,):
                self.optional_decisions[skipped] = {
                    "status": "skipped",
                    "scope": self._scope(skipped, working),
                }
            self.offered_options = None
            self.finished = False
            result = (
                self.last_result
                if self.state.get("pending_request") is None
                and self.last_result
                and self.last_result.get("data", {}).get("ready_for_downstream")
                else {"status": "needs_input", "data": {}}
            )
            output = self._reply(result)
            output["input"] = message
            return output
        options = offered["options"]
        if CONFIRM.fullmatch(message):
            output = self._reply({"status": "needs_input", "data": {}})
            output["reply"] = (
                "Please choose options or say skip before confirming the summary.\n\n"
                + output["reply"]
            )
            output["input"] = message
            return output
        numeric = re.fullmatch(
            r"(?:(?:i\s+)?(?:choose|select|want|prefer)\s+)?(?:options?\s+)?"
            r"\d+(?:\s*(?:,|and|&|、|和)\s*\d+)*[.!]?",
            message,
            re.I,
        )
        selected = []
        if numeric:
            positions = [int(n) for n in re.findall(r"\d+", message)]
            if all(1 <= n <= len(options) for n in positions):
                selected = [options[n - 1]["value"] for n in positions]
            else:
                output = self._reply({"status": "needs_input", "data": {}})
                output["reply"] = (
                    "Please choose a displayed number, or say skip.\n\n" + output["reply"]
                )
                output["input"] = message
                return output
        else:
            remaining = message
            for option in options:
                terms = [option["value"], option["label"], *option.get("aliases", [])]
                pattern = (
                    r"(?<!\w)(?:"
                    + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
                    + r")(?!\w)"
                )
                if re.search(pattern, remaining, re.I):
                    selected.append(option["value"])
                    remaining = re.sub(pattern, " ", remaining, flags=re.I)
            remaining = re.sub(
                r"\b(?:i|would|like|choose|select|want|prefer|and|or|option|options|the|please|to|include)\b",
                " ",
                remaining,
                flags=re.I,
            )
            if re.sub(r"[\s,.!&、和]+", "", remaining):
                return None
        if not selected:
            return None
        values = list(dict.fromkeys(selected))
        self.state["pending_request"] = apply_patch(
            TripShoppingRequest.model_validate(working),
            TripShoppingRequestPatch(set={field: values}),
        ).model_dump(mode="json")
        self.optional_decisions[field] = {
            "status": "selected",
            "scope": self._scope(field, working),
        }
        self.offered_options = None
        self.finished = False
        output = self._reply(
            {
                "status": "needs_input",
                "data": {
                    "state": self.state,
                    "proposed_request": self.state["pending_request"],
                    "selection": {"field": field, "values": values, "offered_options": offered},
                },
            }
        )
        output["input"] = message
        return output

    def _reply(self, result):
        data = result.get("data", {})
        request = self.state.get("pending_request") or self.state["current_request"]
        missing = required_questions(TripShoppingRequest.model_validate(request))
        question = ""
        self.awaiting_field = None
        summary = facts_text(request)
        menu_lines = []
        offered = None
        if (
            not missing
            and not data.get("validation_errors")
            and result["status"] != "error"
            and not data.get("dialogue_action")
            and not data.get("date_clarification")
        ):
            offered = self._next_optional(request, data.get("optional_suggestions"))
        self.offered_options = offered
        if data.get("date_clarification"):
            reference = ("I have recorded " + summary + ". ") if summary else ""
            if data.get("validation_errors"):
                reference += "Your earlier details are still saved. "
            question = data["date_clarification"]
            self.awaiting_field = data.get("date_awaiting_field")
            if "calendar date is invalid" in question and self.awaiting_field in QUESTIONS:
                question += " " + QUESTIONS[self.awaiting_field]
        elif data.get("dialogue_action") == "finish":
            reference = "All set. Your confirmed trip details are saved: " + summary + ". "
            reference += "You can return to add preferences or change these details later."
        elif data.get("dialogue_action") == "correct_review":
            reference = "Your saved details have not been changed. "
            question = "Which trip detail would you like to correct?"
        elif data.get("validation_errors"):
            reference = "I could not record that change. Your earlier details are still saved. "
            reference += "Please clarify only the detail you want to add or change."
            if missing:
                self.awaiting_field = missing[0]["field"]
                question = QUESTIONS[self.awaiting_field]
        elif result["status"] == "error":
            reference = "Your trip details are saved, but a provider is unavailable. "
            reference += "I cannot complete that query right now."
        elif missing:
            self.awaiting_field = missing[0]["field"]
            question = QUESTIONS[self.awaiting_field]
            reference = ("I have recorded " + summary + ". ") if summary else ""
        elif offered:
            field = offered["field"]
            self.awaiting_field = "optional_" + field
            if field == "activities":
                contextual = offered["options"][0]["scope"] == "destination"
                reference = (
                    f"For {request['destination']}, here are optional activity ideas: "
                    if contextual
                    else "Here are general activity ideas; local availability is unverified: "
                )
            else:
                reference = (
                    "Here are optional aesthetic directions; "
                    "these do not replace venue dress codes: "
                )
            for i, option in enumerate(offered["options"], 1):
                detail = ""
                if field == "activities" and option["examples"]:
                    detail = " (for example, " + option["examples"][0]["name"] + ")"
                elif field == "style":
                    detail = " — " + option["description"]
                menu_lines.append(f"{i}) {option['label']}{detail}")
            reference += "\n" + "\n".join(menu_lines) + "\n"
            question = (
                f"Which {'activities' if field == 'activities' else 'styles'} would you like? "
                "Reply with numbers or names, or say skip."
            )
        elif self.state.get("pending_request") is not None:
            reference = "Here are your accumulated trip details: " + summary + ". "
            question = "Are these details correct? Say confirm, or tell me what to change."
        elif data.get("ready_for_downstream"):
            reference = "Your trip details are confirmed: " + summary + ". "
            reference += "I have processed the available travel context. "
            if not request["activities"]:
                reference += (
                    "No activities were specified, so the activity knowledge results are empty. "
                )
            else:
                reference += (
                    "Activity knowledge results are available in the saved reasoning output. "
                )
            weather = data.get("enriched_trip_context", {}).get("weather_context", {})
            if weather.get("status") != "ok":
                reference += "Weather information is unavailable. "
            elif weather.get("data_type") == "historical_climatology":
                reference += "Weather context uses historical climate reference data. "
            reference += "These results do not verify a complete outfit. "
            question = "Would you like to add any optional preferences or change a trip detail?"
        elif data.get("questions"):
            question = data["questions"][0]["question"]
            reference = ("Your trip details are saved: " + summary + ". ") if summary else ""
        else:
            reference = result.get("reason") or "Your trip details are saved."
        reference = (reference + " " + question).strip()
        ui_only = self.settings.get("structured_ui", False) and (
            offered is not None
            or (
                not missing
                and self.state.get("pending_request") is not None
                and not data.get("validation_errors")
                and not data.get("dialogue_action")
                and not data.get("date_clarification")
                and result["status"] != "error"
            )
        )
        if ui_only:
            reply, wording = "", {"provider": "structured_ui", "used_fallback": False}
        else:
            reply, wording = self.responder.respond(reference, question)
        if not ui_only and menu_lines and any(line not in reply for line in menu_lines):
            reply = reference
            wording["used_fallback"] = True
        if offered:
            result = deepcopy(result)
            result.setdefault("data", {})["optional_offer"] = offered
        return {
            "reply": reply,
            "result": result,
            "wording": wording,
            "checkpoint": self.checkpoint(),
        }

    def opening(self):
        if self.finished:
            return self._reply({"status": "ok", "data": {"dialogue_action": "finish"}})
        if self.last_result and self.last_result.get("data", {}).get("ready_for_downstream"):
            return self._reply(self.last_result)
        return self._reply({"status": "needs_input", "data": {}})

    def turn(self, message):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Please enter a nonempty message")
        message = message.strip()
        if GREETING.fullmatch(message):
            output = self.opening()
            output["input"] = message
            return output
        if message.casefold() in ("/suggest", "suggest activities", "suggest styles"):
            self.finished = False
            self.optional_decisions = {}
            output = self._reply(self.last_result or {"status": "needs_input", "data": {}})
            output["input"] = message
            return output
        optional_output = self._optional_turn(message)
        if optional_output is not None:
            return optional_output
        if DECLINE.fullmatch(message):
            pending = self.state.get("pending_request")
            working = pending or self.state["current_request"]
            missing = required_questions(TripShoppingRequest.model_validate(working))
            if not missing:
                if pending is None:
                    self.finished = True
                    result = {"status": "ok", "data": {"dialogue_action": "finish"}}
                else:
                    result = {
                        "status": "needs_input",
                        "data": {"dialogue_action": "correct_review"},
                    }
                output = self._reply(result)
                output["input"] = message
                return output
            output = self._reply({"status": "needs_input", "data": {}})
            output["input"] = message
            return output
        self.finished = False
        date_question = None
        date_awaiting = self.awaiting_field
        if self.date_question and CONFIRM.fullmatch(message):
            output = self._reply(
                {
                    "status": "needs_input",
                    "data": {
                        "date_clarification": self.date_question,
                        "date_awaiting_field": date_awaiting,
                    },
                }
            )
            output["input"] = message
            return output
        if CONFIRM.fullmatch(message):
            if self.state.get("pending_request") is not None:
                payload = {"accept_extraction": True}
            else:
                return self.opening()
        else:
            working = self.state.get("pending_request") or self.state["current_request"]
            date_source = message
            if self.pending_date_source and re.fullmatch(r"\d{4}年?[.!]?", message):
                year = re.match(r"\d{4}", message).group()
                date_source = re.sub(
                    r"(?<!年)(?<!\d)(\d{1,2}月\d{1,2}日?)",
                    lambda m: year + "年" + m.group(),
                    self.pending_date_source,
                )
                from travel_agent.understanding.dates import MONTH_PATTERN

                date_source = re.sub(
                    rf"\b((?:{MONTH_PATTERN})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?)\b"
                    r"(?!\s*,?\s*\d{4})",
                    lambda m: m.group() + ", " + year,
                    date_source,
                    flags=re.I,
                )
                date_source += ". " + message
            effective, date_facts, date_question = normalize_dates(
                date_source, awaiting=self.awaiting_field, current=working
            )
            if date_question and "What is your return date?" in date_question:
                date_awaiting = "end_date"
            if date_source != message:
                # A year-only answer completes the saved dates without replaying
                # stale budgets or preferences from that earlier message.
                effective = (
                    message
                    + ". "
                    + ". ".join(
                        ("Start date" if field == "start_date" else "End date") + " is " + value
                        for field, value in date_facts.items()
                    )
                )
            if date_question and "Which year" in date_question:
                self.pending_date_source = date_source
            elif date_facts:
                self.pending_date_source = None
            elif self.date_question and not date_question:
                date_question = self.date_question
            self.date_question = date_question
            # A short answer has an explicit slot because the assistant just asked it.
            if (
                not date_facts
                and not date_question
                and self.awaiting_field in ("start_date", "end_date")
                and re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}|"
                    r"(?:January|February|March|April|May|June|July|August|September|October|"
                    r"November|December)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}",
                    message,
                    re.I,
                )
            ):
                label = "Start date" if self.awaiting_field == "start_date" else "End date"
                effective = f"{label} is {message}"
            elif (
                not date_facts
                and not date_question
                and self.awaiting_field == "destination"
                and re.fullmatch(r"[A-Za-z]+(?:[ -][A-Za-z]+){0,3}", message)
            ):
                effective = f"My destination is {message}."
            elif (
                not date_facts
                and not date_question
                and self.awaiting_field == "budget"
                and re.fullmatch(
                    r"(?:about |around |up to )?[$€£¥]?\s*[+-]?\d+(?:,\d{3})*(?:\.\d+)?"
                    r"(?:\s+[A-Za-z]+){0,3}[.!]?",
                    message,
                    re.I,
                )
            ):
                effective = f"My budget is {message}"
            payload = {"message": effective}
        payload = self.settings | payload | {"state": self.state}
        result = run(payload, extractor=self.extractor).model_dump(mode="json")
        if date_question:
            result.setdefault("data", {}).update(
                date_clarification=date_question, date_awaiting_field=date_awaiting
            )
        self.state = result.get("data", {}).get("state", self.state)
        self.last_result = result
        output = self._reply(result)
        output["input"] = message
        output["effective_message"] = payload.get("message")
        return output
