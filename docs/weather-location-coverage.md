# Weather location coverage and progressive clarification

## Shared coverage: 12 countries, 15 cities

These are explicitly selected city references for travel planning, not a tourism ranking or country-wide climate estimates. The eight Asian country codes match the countries already present in the Place KG, but this registry is separate from that KG and changes no KG facts or approvals.

| Country | Stored cities |
|---|---|
| Singapore | Singapore |
| Japan | Tokyo, Osaka |
| South Korea | Seoul |
| China | Beijing, Shanghai |
| Malaysia | Kuala Lumpur |
| Thailand | Bangkok |
| Vietnam | Hanoi, Ho Chi Minh City |
| Indonesia | Denpasar |
| United Kingdom | London |
| France | Paris |
| Australia | Sydney |
| United States | New York |

`weather_locations.json` contains the explicit city/country selections, English/Chinese aliases, official geocoding IDs, coordinates, elevation, timezone, source URLs and retrieval timestamps. It does not expand the season-location registry or the Place KG. Registered bare city names use these declared defaults; a registered default is not a claim that no other place worldwide shares that name. Qualifiers such as `Tokyo, Japan`, `东京，日本`, or `Paris, France` are supported. An unrecognized qualifier, e.g. `Paris, Texas`, never silently selects the Paris, France default.

All 15 cities have 12 real monthly ERA5 references for **2015-2024**, totalling **180 monthly rows of statistics** and 54,795 daily temperature samples. Three previously acquired city records were reused; 12 new cities were acquired. Requested coordinates and returned grid coordinates/elevation are kept distinct. These remain ten-year historical planning references, not 30-year climate normals, station observations or future weather predictions. See [weather data and semantics](weather-data.md).

## Decision process

1. **Country only:** `Japan`, `日本`, `JP`, `China`, etc. return `needs_location`, `reason_code=country_requires_city`, and locally available city candidates. No forecast or national climate value is invented. Singapore/新加坡 is an explicit registered city-state scope; `Republic of Singapore` still permits clarification rather than implying a country-wide average. Countries outside the 12-country alias list are also not assigned capitals: non-city geocoder results require city clarification.
2. **Known city/qualified city:** use the declared city point and report `registered_city_default`. An explicit confirmed selection uses `caller_confirmed_coordinates` and must remain bound to the unchanged request destination.
3. **Unknown city:** make one bounded geocoding search, at most five candidates. Multiple results, missing results, or non-city results return `needs_location`. Candidate metadata retains country/admin area, coordinates, elevation, timezone and source. A unique populated-city result can support a forecast, but never triggers automatic historical acquisition.
4. **Within forecast coverage:** fetch and validate the real forecast. API failure or incomplete coverage returns unknown weather with source, `forecast_unavailable`, and retry/explicit-historical-mode guidance. History is not silently substituted.
5. **Outside/crossing coverage:** use an exact local historical point record for the whole trip. If absent, return `out_of_range`, `history_not_cached`, and a structured `acquisition` description with the resolved city, fixed baseline years, one-location bound and explicit command. Normal lookup does not download ten years of data.
6. **Database failure:** `historical_database_unavailable` explains checking the configured path/permissions or explicitly repairing the DB. A broken historical DB does not prevent country clarification or a valid near-term forecast. Geocoding/network failure is separately `geocoding_unavailable`, with retry or manual-confirmation guidance. Missing values never become clear skies or zero rain.

The service preserves the user's `trip_request.destination`. For example, `日本` stays `日本` while a confirmed Tokyo selection is stored separately and the weather output says `location_label=Tokyo`. Clarification codes, candidates and next steps are included in the service questions, so unresolved countries/ambiguous locations block downstream readiness through the existing state logic.

## Scope restrictions

There is **no nearest-city reuse**. A confirmed historical point must match a stored coordinate within 0.0001 degrees; this is a conservative point-identity tolerance, not a validated climate-similarity radius. When both elevations are available, a discrepancy over 50 m also blocks reuse/requests correction. This is an additional project scope guard, not a claim of accuracy within 50 m.

Mountain, coastal, attraction and indoor scopes are not accepted as city-weather measurements. Recognizable region/venue names such as Mount Fuji, Bali, Hokkaido or an indoor museum ask for an explicit city reference or an appropriate local source. Bali is not an alias for Denpasar. A user can explicitly select Denpasar as a **regional outdoor reference**, but it does not describe every beach or mountain on Bali. A genuinely populated place whose name contains words such as “Mount” can be explicitly selected using populated-place geocoding metadata.

City outputs retain an outdoor/microclimate limitation even when a trip has POIs or hiking activities. The feature does not predict indoor temperatures, elevation-sensitive mountain conditions or beach-specific weather, and does not change POI or KG evidence. No model relation or assertion was approved.

## One-city acquisition and preserving a teammate's cache

For missing history, inspect the returned `weather_context.acquisition.location`, confirm the intended city, and save that object as `confirmed-weather-city.json`. It must contain the original `destination`, a city `name`, ISO `country_code`, latitude/longitude, IANA `timezone` and `scope: "city"`. Retain geocoding source/ID/elevation metadata when supplied. Then explicitly run:

```bash
uv run python -m travel_agent.understanding.climatology cache-location \
  --location-file confirmed-weather-city.json
```

This downloads **one** fixed 2015-2024 city history, validates dates/units/missing-value semantics, saves the raw response locally and transactionally upserts that point. It preserves unrelated cities and same-name cities in other countries. Failed historical acquisition returns `historical_acquisition_failed` and leaves the DB unchanged; a corrupt existing DB is rejected before the historical API call. Cached lookup subsequently needs no network. The command is explicit acquisition, not a new normal-query background loop.

After pulling the expanded shared dataset, merge it without removing locally acquired cities:

```bash
uv run python -m travel_agent.understanding.climatology import --merge
```

A missing DB is automatically initialized from the shared snapshot. An old existing DB needs the explicit merge; normal queries never silently replace local additions. Plain `import` retains the earlier full-cache replacement behavior, so use `--merge` when preserving additions matters.

To build an expanded shareable snapshot, all 15 curated cities are the default. Existing matching city/baseline/model records are reused, and new requests are serial with a two-second interval. Use `--cities` to bound the selection, and `--refresh` only when intentionally reacquiring existing cities:

```bash
uv run python -m travel_agent.understanding.climatology build \
  --output artifacts/weather/expanded-climatology.json
```

No account, API key, paid subscription or separate server is required. Keep provider attribution/source metadata when sharing; do not share private `.env` files.

## Actual validation

On 2026-10-21, all 15 shared cities passed offline historical lookup with full source semantics. A separate real geocoding search for Lyon, France returned city/district candidates; explicitly selecting Lyon and running `cache-location` created a 16-city isolated validation DB while preserving its 15 baseline cities. A later Lyon lookup succeeded with network disabled. Lyon is **not** counted as shared coverage. Evidence and raw acquisition responses are local under `artifacts/weather/location-expansion/` and `artifacts/weather/raw/`.

Offline regressions cover country aliases, original user facts, selected-country mismatches, qualified Chinese names, unknown/same-name cities, explicit acquisition failures, same-name foreign-city cache keys, point/elevation/scope restrictions, API/DB/network failures, and existing forecast/history routing. Official geocoding parameters and metadata: [Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api). Free non-commercial use and limits: [Open-Meteo terms](https://open-meteo.com/en/terms).
