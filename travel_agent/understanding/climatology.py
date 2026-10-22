"""Versioned monthly historical references in SQLite; acquisition is explicit, never on lookup."""

import argparse
import calendar
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from travel_agent.schemas import ClimateMonth, ClimateReference, WeatherContext
from travel_agent.understanding.weather import get_json
from travel_agent.understanding.weather_locations import (
    NON_CITY,
    REGISTRY,
    country_for,
    matching_cities,
    validate_city_location,
)

SEED = Path(__file__).with_name("weather_climatology.json")
DEFAULT_DB = Path(__file__).resolve().parents[2] / "artifacts/weather/climatology.sqlite3"
VERSION = "open-meteo-era5-monthly-2015-2024-v1"
METHOD = (
    "Calendar-month pooled arithmetic means of daily min/max Celsius temperatures; "
    "wet-day frequency = days with precipitation >= 1 mm / valid days; "
    "snowfall-day frequency = days with snowfall > 0 cm / valid days. "
    "2015-2024 is a ten-year planning reference, not a standard 30-year climate normal."
)
ATTRIBUTION = "Open-Meteo Historical Weather API; Copernicus ERA5 reanalysis; CC BY 4.0."


def database_path():
    return Path(os.environ.get("WEATHER_CLIMATE_DB", DEFAULT_DB)).expanduser().resolve()


def validate_record(record):
    ZoneInfo(record["timezone"])
    if record["model"] != "era5" or record["baseline_end_year"] < record["baseline_start_year"]:
        raise ValueError("Invalid historical model or reference period")
    for key, bound in (("latitude", 90), ("longitude", 180)):
        value = record[key]
        if isinstance(value, bool) or not math.isfinite(value) or not -bound <= value <= bound:
            raise ValueError("Invalid historical coordinates")
    monthly = [ClimateMonth.model_validate(row) for row in record["monthly"]]
    if sorted(m.month for m in monthly) != list(range(1, 13)):
        raise ValueError("A historical city record must contain every month once")
    for m in monthly:
        expected = sum(
            calendar.monthrange(y, m.month)[1]
            for y in range(record["baseline_start_year"], record["baseline_end_year"] + 1)
        )
        if m.sample_days != expected:
            raise ValueError("Incomplete historical temperature samples")
    ClimateReference.model_validate(
        {key: record[key] for key in ClimateReference.model_fields if key in record}
        | {"location_basis": "registered_city_default"}
    )
    return record


def import_seed(seed=SEED, db=None):
    """Atomically replace a generated cache, never modify the shareable source dataset."""
    db = Path(db or database_path())
    raw = Path(seed).read_bytes()
    data = json.loads(raw)
    if data["schema_version"] != 1 or not data["records"]:
        raise ValueError("Unsupported or empty climate dataset")
    records = [validate_record(r) for r in data["records"]]
    names = [r["name"].casefold() for r in records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate climate city")
    db.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="climate-", suffix=".sqlite3", dir=db.parent)
    os.close(fd)
    try:
        with sqlite3.connect(temporary) as con:
            con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            con.execute(
                "INSERT INTO metadata VALUES ('seed_sha256', ?)", (hashlib.sha256(raw).hexdigest(),)
            )
            con.execute("CREATE TABLE cities (name TEXT PRIMARY KEY, record_json TEXT NOT NULL)")
            con.executemany(
                "INSERT INTO cities VALUES (?, ?)",
                [(r["name"], json.dumps(r, ensure_ascii=False)) for r in records],
            )
        os.replace(temporary, db)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return db


def cache_records(records, db=None):
    """Transactional point-specific upserts; keep all unrelated cities and local additions."""
    records = [validate_record(r) for r in records]
    store = ClimateDatabase(db=db)
    store.records()
    with sqlite3.connect(store.db) as con:
        existing = [
            (key, json.loads(raw))
            for key, raw in con.execute("SELECT name,record_json FROM cities")
        ]
        for record in records:
            key = (
                f"point:{record['country_code']}:{record['latitude']:.6f}:{record['longitude']:.6f}"
            )
            for existing_key, value in existing:
                if (
                    value["name"] == record["name"]
                    and value["country_code"] == record["country_code"]
                    and value["latitude"] == record["latitude"]
                    and value["longitude"] == record["longitude"]
                ):
                    key = existing_key
            con.execute(
                "INSERT INTO cities VALUES (?,?) ON CONFLICT(name) "
                "DO UPDATE SET record_json=excluded.record_json",
                (key, json.dumps(record, ensure_ascii=False)),
            )
    return store.db


class ClimateDatabase:
    def __init__(self, db=None, seed=SEED):
        self.db = Path(db or database_path()).expanduser().resolve()
        self.seed = Path(seed)

    def records(self):
        if not self.db.exists():
            import_seed(self.seed, self.db)
        with sqlite3.connect(self.db.as_uri() + "?mode=ro", uri=True) as con:
            return [
                validate_record(json.loads(row[0]))
                for row in con.execute("SELECT record_json FROM cities")
            ]

    def locate(self, destination, location=None):
        records = self.records()
        matches = matching_cities(destination or "", records) if location is None else records
        if location is not None:
            if (
                location.get("destination") != destination
                or location.get("scope", "city") != "city"
            ):
                return None
            matches = [
                r
                for r in matches
                if (
                    not location.get("country_code")
                    or location["country_code"] == r["country_code"]
                )
                and all(
                    math.isfinite(float(location[k])) and abs(float(location[k]) - r[k]) <= 0.0001
                    for k in ("latitude", "longitude")
                )
                and (
                    location.get("elevation") is None
                    or r.get("elevation") is None
                    or abs(float(location["elevation"]) - r["elevation"]) <= 50
                )
            ]
        if len(matches) != 1:
            return None
        record = matches[0]
        return record

    def fetch(self, request, location=None, routing_reason="outside_forecast_window"):
        try:
            if not request.destination or not request.start_date or not request.end_date:
                return WeatherContext(reason="Historical reference needs a destination and dates")
            if location is not None and location.get("destination") != request.destination:
                return WeatherContext(
                    status="needs_location",
                    reason="Confirmed coordinates belong to another destination",
                )
            record = self.locate(request.destination, location)
            if record is None:
                return WeatherContext(
                    status="out_of_range",
                    routing_reason=routing_reason,
                    reason_code="history_not_cached",
                    reason="No local historical reference for this city/coordinate scope",
                    next_steps=[
                        "For a shared-snapshot city, run climatology import --merge.",
                        "Otherwise confirm a city and run cache-location once. "
                        "Normal queries never download ten-year histories.",
                    ],
                    acquisition={
                        "operation": "cache-location",
                        "max_locations": 1,
                        "baseline_start_year": 2015,
                        "baseline_end_year": 2024,
                        "requires_explicit_command": True,
                        "command": [
                            "python",
                            "-m",
                            "travel_agent.understanding.climatology",
                            "cache-location",
                            "--location-file",
                            "confirmed-weather-city.json",
                        ],
                        "location": location,
                    }
                    if location
                    else None,
                )
            months = []
            cursor = date(request.start_date.year, request.start_date.month, 1)
            end = date(request.end_date.year, request.end_date.month, 1)
            while cursor <= end and len(months) < 12:
                if cursor.month not in months:
                    months.append(cursor.month)
                cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
            reference = {key: record[key] for key in ClimateReference.model_fields if key in record}
            reference["monthly"] = [
                next(m for m in record["monthly"] if m["month"] == month) for month in months
            ]
            reference["location_basis"] = (
                location.get("location_basis", "caller_confirmed_coordinates")
                if location
                else "registered_city_default"
            )
            reference["limitations"] = [
                "Historical monthly averages, not forecasts for the requested dates or extremes.",
                "Wet/snowfall frequencies describe past days, not future probabilities.",
                "Ten-year reanalysis reference; local microclimates and future anomalies "
                "are not captured.",
                "Whole-trip historical mode: no forecast and historical values are blended.",
            ]
            return WeatherContext(
                status="ok",
                data_type="historical_climatology",
                destination=request.destination,
                start_date=request.start_date,
                end_date=request.end_date,
                latitude=record["latitude"],
                longitude=record["longitude"],
                location_label=record["name"],
                location_basis=reference["location_basis"],
                source=record["source_urls"][0],
                retrieved_at=record["updated_at"],
                climate_reference=reference,
                routing_reason=routing_reason,
                reason="Historical monthly planning reference only; not a trip-date forecast",
            )
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            return WeatherContext(
                routing_reason=routing_reason,
                reason_code="historical_database_unavailable",
                reason=f"Historical database unavailable: {type(exc).__name__}",
                next_steps=[
                    "Check WEATHER_CLIMATE_DB and permissions; explicitly import to repair."
                ],
            )


def aggregate(city, payload, source, start_year, end_year):
    expected_units = {
        "temperature_2m_min": "°C",
        "temperature_2m_max": "°C",
        "precipitation_sum": "mm",
        "snowfall_sum": "cm",
    }
    daily = payload["daily"]
    if any(payload["daily_units"].get(k) != v for k, v in expected_units.items()):
        raise ValueError("Unexpected historical units")
    first, last = date(start_year, 1, 1), date(end_year, 12, 31)
    dates = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    if daily["time"] != [d.isoformat() for d in dates]:
        raise ValueError("Historical dates are missing, duplicated or reordered")
    for key in expected_units:
        values = daily[key]
        if len(values) != len(dates):
            raise ValueError("Incomplete historical attributes")
        for value in values:
            if value is None and key in ("precipitation_sum", "snowfall_sum"):
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Missing/non-finite historical temperature or attribute")
            if key in ("precipitation_sum", "snowfall_sum") and value < 0:
                raise ValueError("Negative precipitation/snowfall")
    groups = defaultdict(list)
    for index, day in enumerate(dates):
        if daily["temperature_2m_min"][index] > daily["temperature_2m_max"][index]:
            raise ValueError("Reversed historical daily temperatures")
        groups[day.month].append(index)
    monthly = []
    for month, indices in sorted(groups.items()):

        def frequency(key, threshold):
            values = [daily[key][i] for i in indices]
            # Partial samples stay unknown; never fill missing precipitation or snow with zero.
            if any(v is None for v in values):
                return None
            return round(
                mean(
                    v >= threshold if key == "precipitation_sum" else v > threshold for v in values
                ),
                6,
            )

        monthly.append(
            {
                "month": month,
                "sample_days": len(indices),
                "mean_daily_min_celsius": round(
                    mean(daily["temperature_2m_min"][i] for i in indices), 3
                ),
                "mean_daily_max_celsius": round(
                    mean(daily["temperature_2m_max"][i] for i in indices), 3
                ),
                "wet_day_frequency": frequency("precipitation_sum", 1),
                "snowfall_day_frequency": frequency("snowfall_sum", 0),
            }
        )
    return validate_record(
        {
            "name": city["name"],
            "aliases": city["aliases"],
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "country_code": city["country_code"],
            "elevation": city.get("elevation"),
            "location_id": city.get("id"),
            "timezone": payload["timezone"],
            "grid_latitude": payload["latitude"],
            "grid_longitude": payload["longitude"],
            "grid_elevation_meters": payload.get("elevation"),
            "baseline_start_year": start_year,
            "baseline_end_year": end_year,
            "model": "era5",
            "method": METHOD,
            "dataset_version": VERSION,
            "monthly": monthly,
            "source_urls": [source, *([city["source"]] if city.get("source") else [])],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "attribution": ATTRIBUTION,
            "raw_response_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="Import the committed monthly snapshot; no network")
    imp.add_argument("--seed", type=Path, default=SEED)
    imp.add_argument("--db", type=Path, default=database_path())
    imp.add_argument(
        "--merge", action="store_true", help="Keep unrelated locally acquired city records"
    )
    build = sub.add_parser("build", help="Explicitly acquire real ERA5 historical data")
    build.add_argument("--cities", nargs="+", help="Exact registry city names; defaults to all 15")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--raw-dir", type=Path, default=DEFAULT_DB.parent / "raw")
    build.add_argument("--reuse-from", type=Path, default=SEED)
    build.add_argument("--request-interval", type=float, default=2)
    build.add_argument(
        "--refresh", action="store_true", help="Explicitly reacquire selected cities"
    )
    cache = sub.add_parser("cache-location", help="Explicitly fetch one confirmed city's history")
    cache.add_argument("--location-file", type=Path, required=True)
    cache.add_argument("--db", type=Path, default=database_path())
    cache.add_argument("--raw-dir", type=Path, default=DEFAULT_DB.parent / "raw")
    args = parser.parse_args()
    if args.command == "import":
        if args.merge and args.db.exists():
            data = json.loads(args.seed.read_text())
            if data["schema_version"] != 1:
                parser.error("Unsupported climate dataset schema")
            print(cache_records(data["records"], args.db))
        else:
            print(import_seed(args.seed, args.db))
        return
    if args.command == "cache-location":
        location = json.loads(args.location_file.read_text())
        validate_city_location(location)
        if not location.get("destination") or (
            NON_CITY.search(location["name"])
            and not location.get("feature_code", "").startswith("PPL")
        ):
            parser.error("Confirm a named city bound to the request, not a country/venue")
        country = country_for(location["name"])
        if country and not country.get("city_state"):
            parser.error("A country name is not a city weather location")
        city = dict(location, aliases=[])
        if args.db.exists():
            try:
                ClimateDatabase(db=args.db).records()
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                parser.exit(
                    1,
                    json.dumps(
                        {
                            "status": "error",
                            "reason_code": "historical_database_unavailable",
                            "reason": type(exc).__name__,
                            "next_steps": [
                                "Repair the local DB before acquiring history; no API call made."
                            ],
                        }
                    )
                    + "\n",
                )
        # At most one historical API request; failed acquisition makes no DB changes.
        try:
            record = acquire_city(city, args.raw_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.exit(
                1,
                json.dumps(
                    {
                        "status": "error",
                        "reason_code": "historical_acquisition_failed",
                        "reason": type(exc).__name__,
                        "http_status": getattr(exc, "code", None),
                        "source": getattr(
                            exc, "url", "https://archive-api.open-meteo.com/v1/archive"
                        ),
                        "next_steps": [
                            "Retry when the historical API is available; cache unchanged."
                        ],
                    }
                )
                + "\n",
            )
        cache_records([record], args.db)
        print(f"Cached {record['name']} ({record['country_code']}); other records preserved")
        return
    registry = json.loads(REGISTRY.read_text())["records"]
    names = args.cities or [r["name"] for r in registry]
    selected = [r for name in names for r in registry if r["name"] == name]
    if len(selected) != len(names) or len(set(names)) != len(names):
        parser.error("Choose unique exact city names from weather_locations.json")
    records = []
    reusable = (
        json.loads(args.reuse_from.read_text())["records"] if args.reuse_from.exists() else []
    )
    if args.request_interval < 0:
        parser.error("request-interval cannot be negative")
    acquired = 0
    for city in selected:
        previous = next(
            (
                r
                for r in reusable
                if r["name"] == city["name"]
                and r["country_code"] == city["country_code"]
                and r["latitude"] == city["latitude"]
                and r["longitude"] == city["longitude"]
                and r["baseline_start_year"] == 2015
                and r["baseline_end_year"] == 2024
                and r["model"] == "era5"
            ),
            None,
        )
        if args.refresh:
            previous = None
        if previous:
            record = dict(
                validate_record(previous),
                elevation=city.get("elevation"),
                location_id=city.get("id"),
            )
        else:
            if acquired:
                time.sleep(args.request_interval)
            record = acquire_city(city, args.raw_dir)
            acquired += 1
        records.append(record)
        print(
            f"{'Reused' if previous else 'Acquired'} {city['name']}: "
            "12 monthly references (2015-2024)",
            flush=True,
        )
    # Keep the previous dataset intact if any acquisition/validation failed.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "records": records}, indent=2) + "\n")


def acquire_city(city, raw_dir, transport=get_json):
    query = urlencode(
        {
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "start_date": "2015-01-01",
            "end_date": "2024-12-31",
            "timezone": "auto",
            "models": "era5",
            "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,snowfall_sum",
        }
    )
    source = "https://archive-api.open-meteo.com/v1/archive?" + query
    payload = transport(source)
    record = aggregate(city, payload, source, 2015, 2024)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    point = hashlib.sha256(f"{city['latitude']},{city['longitude']}".encode()).hexdigest()[:12]
    (raw_dir / f"{city['country_code']}-{point}.json").write_text(json.dumps(payload))
    return record


if __name__ == "__main__":
    main()
