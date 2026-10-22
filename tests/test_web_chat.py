"""Chat clarifications preserve explicit requirements without guessing choices."""

from datetime import date

from travel_agent.web.chat import contextual_answer, merge_candidate, next_question

TODAY = date(2026, 10, 21)


def trip(**updates):
    return {
        "destination": "Tokyo",
        "start_date": "2026-10-23",
        "end_date": "2026-10-25",
        "activities": ["sightseeing", "dinner"],
        "budget": 500,
        "currency": "SGD",
        "style": ["minimalist"],
        "dress_codes": {"dinner": "smart casual"},
        **updates,
    }


def test_questions_only_require_destination_dates_and_budget():
    assert next_question({}, today=TODAY)[0] == "destination"
    assert next_question(trip(destination="Japan"), today=TODAY)[0] == "destination"
    for updates, expected in [
        ({"start_date": None}, "dates"),
        ({"budget": None}, "budget"),
    ]:
        assert next_question(trip(**updates), today=TODAY)[0] == expected
    assert next_question(
        trip(end_date="2026-12-25", activities=[], style=[], dress_codes={}), today=TODAY
    ) == (None, None)
    assert next_question(trip(), today=TODAY) == (None, None)
    assert next_question(trip(style=[]), style_declined=True, today=TODAY) == (None, None)


def test_bare_choices_are_bound_to_the_asked_activity_only():
    request = trip(activities=["dinner", "business meeting"], dress_codes={})
    assert contextual_answer("smart casual", "dress_codes.dinner", request) == (
        "Dinner dress code is smart casual."
    )
    assert contextual_answer("smart casual", "dress_codes.business meeting", request) == (
        "smart casual"
    )
    assert contextual_answer("500 SGD.", "budget", request) == (
        "My total clothing budget is 500 SGD."
    )
    assert contextual_answer("500", "budget", request) == "500"
    assert contextual_answer("yes", "dress_codes.dinner", request) == "yes"
    assert contextual_answer("minimalist", "style", request) == "I prefer minimalist outfits."


def test_followup_patch_keeps_existing_trip_and_does_not_mutate_previous():
    previous = merge_candidate(trip(budget=None, currency=None), {})
    updated = merge_candidate({"set": {"budget": 500, "currency": "SGD"}}, previous)
    assert updated["destination"] == "Tokyo"
    assert updated["activities"] == ["sightseeing", "dinner"]
    assert updated["dress_codes"] == {"dinner": "smart casual"}
    assert updated["budget"] == 500
    assert previous["budget"] is None


def test_bound_answers_pass_the_existing_source_grounder():
    from travel_agent.schemas import TripShoppingRequest
    from travel_agent.understanding.extraction import ground_extraction

    previous = TripShoppingRequest.model_validate(trip(dress_codes={}))
    for message, field, expected in [
        ("smart casual", "dress_codes.dinner", {"dress_codes": {"dinner": "smart casual"}}),
        ("500 SGD", "budget", {"budget": 500, "currency": "SGD"}),
    ]:
        effective = contextual_answer(message, field, previous.model_dump(mode="json"))
        grounded, _ = ground_extraction({}, effective, True, previous)
        assert grounded["set"] == expected
    effective = contextual_answer("hiking and museums", "activities", {})
    grounded, _ = ground_extraction({}, effective, True, previous)
    assert set(grounded["set"]["activities"]) == {"hiking", "museum visit"}
    effective = contextual_answer("Los Angeles", "destination", {})
    grounded, _ = ground_extraction({"destination": "Los Angeles"}, effective, True, previous)
    assert grounded["set"]["destination"] == "Los Angeles"
    effective = contextual_answer("no preference", "style", previous.model_dump(mode="json"))
    grounded, _ = ground_extraction({}, effective, True, previous)
    assert grounded["remove"] == {"style": []}
