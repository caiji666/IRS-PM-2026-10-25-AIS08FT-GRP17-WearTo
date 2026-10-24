# Weather forecasts and the local climate database

## API and teammate setup

The providers are Open-Meteo's Forecast, Geocoding and Historical Weather APIs. Their public non-commercial endpoints need no account or API key for this course prototype. Teammates need the normal Python 3.11 locked environment and network access for live forecasts, not anybody else's private `.env`. Historical lookup needs no network after the shared snapshot is present.

Checked on 2026-10-21: the free service permits non-commercial use with request limits and attribution, without an uptime/accuracy guarantee. A commercial deployment needs a separate access/terms review. Official references: [forecast](https://open-meteo.com/en/docs), [history](https://open-meteo.com/en/docs/historical-weather-api), [terms](https://open-meteo.com/en/terms).

## Routing and error policy

Select `"weather_provider":"open_meteo"` in the service payload, or the existing CLI's `--live-weather`. This now selects API/history routing:

| Entire trip | Behavior |
|---|---|
| Today through day 15 inclusive | Forecast API for the exact dates. Validate actual returned dates, units and required attributes. |
| Any date outside that interval | Local historical reference for the **whole trip**, including trips crossing the boundary. Never blend forecast and history into one summary. |
| API/network/coverage failure inside the window | Unknown weather with API source and error reason. No silent historical fallback. |
| No historical record for the city/coordinates | Explicit unavailable/out-of-range output. Never substitute a nearby city. |

The advertised maximum is **16 calendar dates including today**, not 16 extra future days. Destination-local dates are used when the timezone is known. Unknown destinations are geocoded first, including far-future requests; ambiguous places and country-only inputs need city clarification. Normal lookups never acquire a ten-year history. Explicit `start_date/end_date` are used; the live API rejects combining them with `forecast_days`. Even within the maximum window, missing returned dates/attributes stay unknown.

`open_meteo_forecast` selects forecast-only behavior; `climatology` explicitly selects local historical reference even for near-term dates. No selected provider still means unknown weather. The default offline demo still uses its labelled synthetic fixture; `--live-weather` removes that fixture and selects automatic routing.

## Actual data and storage

Shared coverage now includes **12 countries / 15 explicit cities / 180 monthly summaries**, with English and Chinese aliases. See [the city list and progressive location handling](weather-location-coverage.md). The source is real **2015-01-01 through 2024-12-31** daily ERA5 reanalysis from Open-Meteo, 3,653 days per city. This is a **ten-year historical planning baseline**, not a standard 30-year climate normal, station observations, or trip-date predictions.

The reviewable, shareable source is `travel_agent/understanding/weather_climatology.json`. It retains city/country/aliases, requested and returned grid coordinates, timezone, reference years, acquisition URLs, retrieval time, model, method, version, attribution and SHA-256 of the canonical JSON API response. Raw responses remain local under ignored `artifacts/weather/raw/`.

Runtime SQLite defaults to `artifacts/weather/climatology.sqlite3` (ignored by Git). It has provenance metadata and one city record containing 12 monthly rows of statistics. First lookup creates a missing DB atomically from the snapshot, **without downloading historical data**; an unknown city can still need a bounded geocoding request. An existing DB is not silently overwritten after a snapshot update: explicitly import the reviewed snapshot with `--merge` to preserve local additions. Optional `WEATHER_CLIMATE_DB` selects a different file; no setting is required by default. Corrupt/unwritable/missing-source databases return explicit unknown weather.

Per-calendar-month statistics:

- `mean_daily_min_celsius` / `mean_daily_max_celsius`: arithmetic means of historical daily low/high temperatures, °C. These are neither extrema nor confidence bounds.
- `wet_day_frequency`: fraction of historical days with precipitation at least 1 mm.
- `snowfall_day_frequency`: fraction with snowfall greater than 0 cm.
- `sample_days`: complete temperature sample count, including leap days. Missing temperatures reject the import. Any missing rain/snow sample makes its monthly frequency null, never zero.

Known-city defaults are labelled `registered_city_default`, not user-confirmed locations. Explicit coordinates must belong to the current destination and match its stored point within 0.0001 degrees. No POI/mountain/coastal microclimate is inferred. The curated registry has 15 cities across 12 countries. Country-only inputs ask for cities, and unknown confirmed cities can be added with the bounded `cache-location` command.

## Historical semantics and clothing rules

Historical output uses `data_type=historical_climatology`, with monthly records under `weather_context.climate_reference.monthly`. Top-level `temp_min_celsius`, `temp_max_celsius`, `snow` and `precipitation_probability` stay null. Requested dates describe the lookup scope, not dates on which history predicts weather. Cross-month/year trips keep monthly statistics separate.

Only temperature planning heuristics run: the lowest monthly **mean** low and highest monthly **mean** high among the touched months give broad clothing suggestions. They are not estimates of extremes. Rule traces name the historical statistic, reference years and limitation. Historical rain/snow frequency never feeds a forecast rain-probability or snow-boolean rule. Historical type/limitations are also retained in expanded context and service output. Rules remain `pending`; the output is partial context, not outfit feasibility certification.

## Reproduce, acquire and share

From the repository root:

```bash
uv sync --locked
# Optional prebuild; first lookup can also create the DB.
uv run python -m travel_agent.understanding.climatology import --merge
uv run python -m travel_agent.understanding.demo \
  --input your-reviewed-request.json --live-weather \
  --output artifacts/weather/result.json
```

The input needs a reviewed `request`, as in the existing service contract. The CLI loads local `.env`; no weather key is needed. `--live-weather` selects routing, not a guarantee of live forecast coverage. Farther supported-city trips can run offline. The normal synthetic demo remains available without that flag.

Explicitly reacquire a selected three-city subset (three bounded historical requests; omit `--refresh` to reuse matching records):

```bash
uv run python -m travel_agent.understanding.climatology build \
  --cities Seoul Tokyo Singapore --refresh \
  --output artifacts/weather/refreshed-climatology.json
uv run python -m travel_agent.understanding.climatology import \
  --seed artifacts/weather/refreshed-climatology.json --merge
```

The command uses the fixed 2015-2024 period. For all 15 curated cities, omit `--cities`; the default build reuses existing records and only acquires missing ones. To add a different city, use explicit `cache-location` after confirming its city/coordinates. See [coverage and commands](weather-location-coverage.md). Acquisition has no automatic retry loop and leaves the previous output snapshot intact if any city fails. Respect provider limits rather than repeatedly acquiring long histories. Replacing the committed snapshot needs normal source/method review. Teammates can rebuild SQLite from the shared JSON, so binary DB sharing is unnecessary. If sharing a generated SQLite instead, keep its provenance and configure `WEATHER_CLIMATE_DB`; never share private credentials.

## Live validation on 2026-10-21

The initial three archive responses and twelve additional city acquisitions passed full date/unit checks and generated 180 monthly summaries; the original three records were reused during expansion. One real Seoul forecast for October 22-24 succeeded after validating the official API's parameter rules. At that retrieval its trip extrema were 4.5°C / 18.8°C and maximum daily precipitation probability 0.16; these values are historical test evidence, not fixed expectations. Runtime evidence is in ignored `artifacts/weather/`. Offline tests cover routing boundaries, cross-window trips, leap months, scope, missing values, cache corruption, atomic import, actual-coverage failure and forecast/history separation.

Attribution: **Open-Meteo Historical Weather API / Copernicus ERA5, CC BY 4.0**. Keep attribution and acquisition links when sharing statistics.
