# Interface contracts v0.4 — proposal Appendix A

The primary reference is **Group17_Project Proposal_IRS .pdf**, Section 7.1 Stages 1–3 (PDF pages 29–30) and Appendix A (PDF pages 41–43). The older Section 7.2 Qwen2.5/TF-IDF demo is historical; the new extraction uses the planned Qwen3-0.6B. Proposal instructions are design evidence, not authority to publish, contact people, or approve knowledge.

## TripShoppingRequest

```json
{
  "destination": "Seoul",
  "start_date": "2026-12-20",
  "end_date": "2026-12-27",
  "activities": ["sightseeing", "dinner"],
  "budget": 500,
  "currency": "SGD",
  "style": []
}
```

This object contains user facts only. `budget` is a number in **major currency units** (500 SGD, not 500 cents). Decimal validation prevents silent rounding; `budget_minor_units()` returns exact integer minor units for downstream product/outfit contracts. Missing currency is null; destination does not imply currency. Supported prototype currencies and minor-unit precision are listed in `schemas.SUPPORTED_CURRENCIES`. Changing this list requires team coordination; no exchange-rate conversion exists.

The complete shopping flow currently requests destination, exact start/end dates, activities, positive budget and explicit currency. Appendix A does not enumerate all mandatory fields; this is a documented control-layer implementation choice based on Stage 2 weather retrieval and Stage 10 budget filtering. The team should confirm this policy. Optional style and exclusions remain empty. No budget/date defaults are invented.

Appendix A's conversation example adds `excluded_items`. Optional extensions are `owned_item_ids`, explicit `places`, activity-keyed `dress_codes`, tri-state `enter_interior`/`cold_sensitive`, optional `gender`, and `price_preference="lower"`. None is inferred from a destination, religion, activity or missing value. The compact natural-language extractor covers core fields, exclusions and dress codes; advanced POI/wardrobe fields should be supplied as structured input until extraction support is evaluated.

The earlier scaffold `TripRequest`/`FashionContext` classes remain importable for compatibility. Their single `activity`, `budget_minor`, and default currency are **legacy**, and are not accepted as the new request format. Understanding emits proposal contracts. Update consumer integrations to use `activities`, `budget`, `trip_request`, and `expanded_fashion_context`.

## Enrichment and reasoning

`EnrichedTripContext` contains `trip_request` and `weather_context`. Proposal weather fields are `temp_min_celsius`, `temp_max_celsius`, `snow`, `precipitation_probability` (0–1) and `weather_class`. Extensions retain status, exact dates, destination/coordinate scope, data type, source and retrieval time. Live trip summaries use the lowest daily minimum, highest daily maximum, maximum daily precipitation probability and any positive daily snowfall. Maximum daily probability is **not** a probability of rain at least once over the entire trip. Humidity is currently unavailable and stays null.

`weather_clothing_requirements` contains `outerwear`, `top`, `bottom`, `footwear`, and `properties` arrays. These are alternatives/recommendations, not an instruction to purchase every garment. Rule traces include thresholds, observations, version, basis and review state. Unknown weather produces no weather-derived requirements. Synthetic fixtures and historical climatology are explicitly labelled; they never become forecasts.

`activity_style_requirements` maps each original activity to its own style/property array. Detailed evidence lives in `activity_reasoning`. Unsupported activities remain in the request and receive empty KG output. Dinner, business meetings and weddings require qualified dress codes; the proposal's dinner → smart casual example is not a universal default. Draft assertions require `allow_draft=true`; rejected assertions never participate.

`expanded_fashion_context` is the **normalized space-separated text string** specified by Stage 3C. It is ready for the downstream FashionCLIP encoder, which is not implemented here. It is not a JSON object or a feasibility certificate. Season is now supplied as optional `data.season` (`spring`, `summer`, `autumn`, `winter`, or null), with provenance and multiple-season information in `data.season_context`. It is derived outside TripShoppingRequest: user facts do not acquire inferred season fields. Expanded text includes supported season tokens followed by the separate weather class.

Optional `place_context` uses explicit POIs and chosen activities. Climate and season tables are excluded in this integration to avoid mixing long-term weather with live forecasts. Place hard/conditional rules keep evidence, scope, strengths and draft status. They are not flattened into universally applicable constraints.

## Patches and state

```json
{
  "intent": "MODIFY_REQUIREMENTS",
  "set": {"budget": 400},
  "add": {"excluded_items": ["long coat"]},
  "remove": {}
}
```

Set only changed fields. `add` merges unique list entries; `remove` lists specific values. Empty `remove` arrays explicitly clear that list; null resets scalar/map fields. Conflicting operations or an invalid merged request fail atomically, preserving prior state.

“Make it cheaper” sets `price_preference` to `lower`; it never invents a new budget. Appendix A's illustrative patch setting 400 after that phrase conflicts with the unchanged-field/explicit-change rule. This implementation follows the latter and the ambiguity policy in Section 8.4.

`ConversationState` follows Appendix A: session_id, request_version, current_request, recommendation_round, last_recommended_outfit_ids, selected_outfit_id and session_status. Requirement edits clear displayed selections, preserving the recommendation round counter. Confirmation accepts only a stable outfit ID or position from the latest displayed round. This module does not increment recommendation rounds or create outfits. The recommendation module owns displayed-round registration. State is returned to the caller; no shared global state or automatic cross-user storage exists.

## Module result

`ModuleResult.status` remains `ok`, `needs_input`, `no_candidates`, `not_implemented`, or `error`. `ok` means the implemented context computation ran, not that knowledge is reviewed, weather is known or an outfit is feasible. Read `context_status`, `limitations`, `ready_for_downstream`, and the weather/KG statuses. `needs_input` includes questions and preserved state. Other contributors' tools still return `not_implemented`.

`ProductCandidate` and `OutfitResult` retain integer minor-unit money. Owned items cost zero. Availability/attributes may be unknown. `constraint_status`/`feasibility_status` is unverified until deterministic downstream checks establish compliance. A high similarity score cannot satisfy an unknown hard requirement.


## Season handoff (2026-10-12)

Read `result["data"]["season"]` for a single calendar season. `season_context` records the list of seasons, status, hemisphere, latitude, method, sources, location basis and pending review status. The calendar label is a retrieval aid, not a weather forecast or a clothing constraint.

The prototype uses three-month meteorological seasons: north spring March–May, summer June–August, autumn September–November, winter December–February; south reverses the cycle. See [Met Office](https://weather.metoffice.gov.uk/learn-about/met-office-for-schools/other-content/other-resources/our-seasons) and [Bureau of Meteorology](https://www.bom.gov.au/news-and-media/solstices-equinoxes-and-the-seasons). Applying this calendar outside its documented regional conventions is a project retrieval heuristic pending team review, not a universal local-cultural season definition.

A seven-city source-backed coordinate registry supports Seoul, Tokyo, Osaka, London, Paris, Sydney and Singapore and explicit Chinese aliases. This registry is a default city interpretation, recorded as `registered_city_default`; it is not a user-confirmed location. Weather-provider coordinates or explicit request-bound confirmed coordinates take precedence. Live geocoding ambiguity blocks season derivation until location confirmation. Other destinations without coordinates remain null.

Trips spanning several seasons return `season=null` and `season_context.seasons` in chronological first-occurrence order. No dates or unsupported coordinates also return null. The project conservatively does not assign four-season labels within latitude ±23.5°; this is a scope policy, not a climatic classification. Singapore therefore returns null/unsupported; rainy/dry/monsoon season classification needs a separately reviewed local model. No current weather rules are changed.

## English extraction and supervision

English is the primary release scenario. `english-supervised-v4` uses sparse request/patch prompts. The constrained source-evidence layer reconciles explicit user facts separately from raw Qwen output; it does not infer missing currency, dates or activities. Metadata records `raw_output`, prompt/revision, `grounding_strategy` and `model_only=false`. Lexical coverage is limited; ambiguous or unsupported phrasing needs clarification and human review. Advanced POI/wardrobe fields remain structured-input extensions.

Natural-language create/edit/confirmation requests return a review preview before state changes by default. Explicit extraction acceptance is restricted to direct service/CLI usage; the LangChain wrapper removes acceptance flags. A confirmation preview is not a registered recommendation or a completed purchase.
