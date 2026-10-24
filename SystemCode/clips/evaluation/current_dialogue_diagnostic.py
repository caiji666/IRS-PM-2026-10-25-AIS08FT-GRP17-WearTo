"""Live extraction plus current dialogue controller; reference wording isolates state quality."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from travel_agent.schemas import TripShoppingRequest
from travel_agent.understanding.conversation import Conversation
from travel_agent.understanding.extraction import QwenExtractor

OUT = Path("evaluation/results/2026-10-22")


def request(**values):
    return TripShoppingRequest(**values).model_dump(mode="json")


def step(message, facts, *, confirmed=False, date_question=False):
    return {
        "message": message,
        "expected_request": request(**facts),
        "expected_confirmed": confirmed,
        "expected_date_question": date_question,
    }


def dataset():
    tokyo = {
        "destination": "Tokyo",
        "start_date": "2026-10-28",
        "end_date": "2026-11-02",
        "budget": 500,
        "currency": "SGD",
    }
    seoul = {**tokyo, "destination": "Seoul"}
    return [
        {
            "id": "C01_mixed_abbreviated_dates_and_confirmation",
            "turns": [
                step("Tokyo, 28 Oct 2026 to 2 Nov 2026, budget 500 SGD", tokyo),
                step("skip all", tokyo),
                step("confirm", tokyo, confirmed=True),
            ],
        },
        {
            "id": "C02_slash_dates",
            "turns": [
                step(
                    "Tokyo, 2027/04/03 to 2027/04/08, budget 500 SGD",
                    {**tokyo, "start_date": "2027-04-03", "end_date": "2027-04-08"},
                )
            ],
        },
        {
            "id": "C03_year_only_clarification",
            "turns": [
                step(
                    "Tokyo, Oct 28 to Nov 2, budget 500 SGD",
                    {"destination": "Tokyo", "budget": 500, "currency": "SGD"},
                    date_question=True,
                ),
                step("2027", {**tokyo, "start_date": "2027-10-28", "end_date": "2027-11-02"}),
            ],
        },
        {
            "id": "C04_ambiguous_date_cannot_be_confirmed",
            "turns": [
                step(
                    "Tokyo, 03/04/2027, budget 500 SGD",
                    {"destination": "Tokyo", "budget": 500, "currency": "SGD"},
                    date_question=True,
                ),
                step(
                    "confirm",
                    {"destination": "Tokyo", "budget": 500, "currency": "SGD"},
                    date_question=True,
                ),
            ],
        },
        {
            "id": "C05_incremental_short_answers",
            "turns": [
                step("Tokyo", {"destination": "Tokyo"}),
                step("2026-10-28", {"destination": "Tokyo", "start_date": "2026-10-28"}),
                step(
                    "2026-11-02",
                    {"destination": "Tokyo", "start_date": "2026-10-28", "end_date": "2026-11-02"},
                ),
                step(
                    "500",
                    {
                        "destination": "Tokyo",
                        "start_date": "2026-10-28",
                        "end_date": "2026-11-02",
                        "budget": 500,
                    },
                ),
                step("SGD", tokyo),
                step("skip all", tokyo),
                step("confirm", tokyo, confirmed=True),
            ],
        },
        {
            "id": "C06_explicit_activity_and_skip_style",
            "turns": [
                step("Seoul, 28 Oct 2026 to 2 Nov 2026, budget 500 SGD", seoul),
                step("cafe visit", {**seoul, "activities": ["cafe visit"]}),
                step("skip", {**seoul, "activities": ["cafe visit"]}),
                step("confirm", {**seoul, "activities": ["cafe visit"]}, confirmed=True),
            ],
        },
    ]


class ReferenceResponder:
    def respond(self, reference, question):
        return reference, {"provider": "deterministic_reference_for_evaluation"}


class CountedExtractor(QwenExtractor):
    calls = 0

    def generate_raw(self, *args, **kwargs):
        self.calls += 1
        return super().generate_raw(*args, **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = dataset()
    encoded = json.dumps(cases, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    frozen = OUT / "current-dialogue-cases.json"
    if frozen.exists() and frozen.read_text() != encoded:
        raise ValueError("Preserve previously frozen labels")
    frozen.write_text(encoded)
    report = {
        "label_status": "assistant_authored_pending_human_review",
        "role": "new_synthetic_diagnostic_not_deployment_benchmark",
        "dataset_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "started_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "wording_provider": "deterministic reference; real Qwen used for extraction",
        "scenarios": [],
    }
    extractor = CountedExtractor(local_files_only=True)
    for case in cases:
        chat = Conversation(extractor=extractor, responder=ReferenceResponder())
        turns = []
        for expected in case["turns"]:
            response = chat.turn(expected["message"])
            actual = chat.state.get("pending_request") or chat.state["current_request"]
            diff = {
                k: {"expected": v, "actual": actual.get(k)}
                for k, v in expected["expected_request"].items()
                if actual.get(k) != v
            }
            confirmed = (
                chat.state.get("pending_request") is None
                and chat.state.get("request_version", 0) > 0
            )
            checks = {
                "facts_match": not diff,
                "confirmation_match": confirmed == expected["expected_confirmed"],
                "date_question_match": bool(chat.date_question)
                == expected["expected_date_question"],
            }
            turns.append(
                {
                    "expected": expected,
                    "actual_request": actual,
                    "checks": checks,
                    "passed": all(checks.values()),
                    "field_errors": diff,
                    "response": response,
                }
            )
        row = {"id": case["id"], "turns": turns, "passed": all(t["passed"] for t in turns)}
        report["scenarios"].append(row)
        print(case["id"], row["passed"], flush=True)
        (OUT / "current-dialogue-result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    report["actual_model_calls"] = extractor.calls
    report["finished_at"] = datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
    report["summary"] = {
        "scenarios": len(cases),
        "scenarios_passed": sum(s["passed"] for s in report["scenarios"]),
        "turns": sum(len(s["turns"]) for s in report["scenarios"]),
        "turns_passed": sum(t["passed"] for s in report["scenarios"] for t in s["turns"]),
    }
    (OUT / "current-dialogue-result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(report["summary"], flush=True)


if __name__ == "__main__":
    main()
