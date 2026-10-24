"""Bounded city resolution for weather, not a country climate model or a POI knowledge graph."""

import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from travel_agent.schemas import WeatherContext

REGISTRY = Path(__file__).with_name("weather_locations.json")
NON_CITY = re.compile(
    r"\b(mount|mountain|beach|temple|museum|palace|indoor|ski resort|national park)\b"
    r"|\b(Bali|Hokkaido|Alps)\b|富士山|巴厘岛|北海道|阿尔卑斯|海滩|海滨|博物馆|寺庙|滑雪场|国家公园|室内",
    re.IGNORECASE,
)


def normalize(value):
    return " ".join(
        "".join(
            c
            for c in unicodedata.normalize("NFKD", str(value).casefold())
            if not unicodedata.combining(c)
        ).split()
    )


def load_registry():
    return json.loads(REGISTRY.read_text())


def country_for(value, registry=None):
    for country in (registry or load_registry())["countries"]:
        if normalize(value) in {normalize(n) for n in [country["name"], *country["aliases"]]}:
            return country
    return None


def matching_cities(value, cities, registry=None):
    name, separator, qualifier = value.replace("，", ",").partition(",")
    country = country_for(qualifier.strip(), registry) if separator else None
    if separator and country is None:
        return []  # E.g. Paris, Texas must not become the Paris, France default.
    return [
        city
        for city in cities
        if normalize(name) in {normalize(n) for n in [city["name"], *city.get("aliases", [])]}
        and (country is None or city["country_code"] == country["code"])
    ]


def validate_city_location(location):
    if location.get("scope", "city") != "city":
        raise ValueError("Select a city-level outdoor weather reference, not a venue microclimate")
    if location.get("feature_code") and not location["feature_code"].startswith("PPL"):
        raise ValueError("Geocoding feature is not a populated city")
    for key, bound in (("latitude", 90), ("longitude", 180)):
        value = location[key]
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or not -bound <= float(value) <= bound
        ):
            raise ValueError("Invalid coordinates")
    if not location.get("name") or not location.get("timezone"):
        raise ValueError("City name and IANA timezone are required")
    if NON_CITY.search(location["name"]) and not location.get("feature_code", "").startswith("PPL"):
        raise ValueError("Choose a city name rather than a region, attraction or indoor venue")
    ZoneInfo(location["timezone"])
    if not re.fullmatch(r"[A-Z]{2}", location.get("country_code", "")):
        raise ValueError("City country_code is required")
    if location.get("elevation") is not None and not math.isfinite(float(location["elevation"])):
        raise ValueError("Invalid elevation")
    return location


def candidate(city):
    keys = (
        "name",
        "country",
        "country_code",
        "admin1",
        "latitude",
        "longitude",
        "timezone",
        "elevation",
        "id",
        "feature_code",
        "scope",
        "source",
        "retrieved_at",
    )
    return {key: city[key] for key in keys if key in city}


class WeatherLocationResolver:
    def __init__(self, transport, cached_cities=()):
        self.transport = transport
        self.registry = load_registry()
        self.cities = list(self.registry["records"])
        for record in cached_cities:
            if not any(
                c["name"] == record["name"]
                and c["country_code"] == record["country_code"]
                and c["latitude"] == record["latitude"]
                and c["longitude"] == record["longitude"]
                for c in self.cities
            ):
                self.cities.append(record)

    def resolve(self, destination, confirmed=None):
        country = country_for(destination, self.registry)
        known = matching_cities(destination, self.cities, self.registry)
        # Singapore is already an explicit registered city-state scope, not a national mean.
        if country and confirmed is None and not (country.get("city_state") and known):
            choices = [candidate(c) for c in self.cities if c["country_code"] == country["code"]]
            return WeatherContext(
                status="needs_location",
                destination=destination,
                reason_code="country_requires_city",
                reason="A country has multiple climates; choose a city or region for weather",
                location_candidates=choices,
                next_steps=[
                    "Choose a candidate city or name another city; keep country and city explicit."
                ],
            )
        if confirmed is None and not known and NON_CITY.search(destination):
            return self.scope_question(destination)
        if confirmed is not None:
            if confirmed.get("destination") != destination:
                return WeatherContext(
                    status="needs_location",
                    reason_code="stale_coordinates",
                    reason="Confirmed coordinates belong to another destination",
                    next_steps=["Confirm a location bound to the current destination."],
                )
            if confirmed.get("scope", "city") != "city":
                return self.scope_question(destination)
            selected = dict(confirmed)
            if len(known) == 1:
                # Fill metadata only at the exact registered point; never move coordinates.
                point = known[0]
                try:
                    same = all(
                        abs(float(selected[k]) - point[k]) <= 0.0001
                        for k in ("latitude", "longitude")
                    )
                except (KeyError, ValueError, TypeError):
                    same = False
                if same:
                    selected = point | selected
            try:
                validate_city_location(selected)
            except (ValueError, KeyError, TypeError) as exc:
                return WeatherContext(
                    status="needs_location",
                    reason_code="incomplete_coordinates",
                    reason=str(exc),
                    next_steps=[
                        "Select a geocoding candidate with city, country, timezone and coordinates."
                    ],
                )
            if (
                len(known) == 1
                and same
                and (
                    selected.get("elevation") is not None
                    and point.get("elevation") is not None
                    and abs(float(selected["elevation"]) - point["elevation"]) > 50
                )
            ):
                return WeatherContext(
                    status="needs_location",
                    reason_code="elevation_mismatch",
                    reason="Elevation differs from the registered city by over 50 m",
                    next_steps=[
                        "Confirm the city point or use a mountain-specific weather source."
                    ],
                )
            if country and selected["country_code"] != country["code"]:
                return WeatherContext(
                    status="needs_location",
                    reason_code="country_mismatch",
                    reason="Selected city is outside the requested country",
                    next_steps=[
                        "Select a city in the requested country or correct the destination."
                    ],
                )
            if len(known) == 1 and (
                selected["country_code"] != point["country_code"]
                or not matching_cities(selected["name"], [point], self.registry)
            ):
                return WeatherContext(
                    status="needs_location",
                    reason_code="city_mismatch",
                    reason="Confirmed city does not match the requested registered city",
                    next_steps=[
                        "Correct the destination or confirm the intended city's coordinates."
                    ],
                )
            selected.update(
                destination=destination, scope="city", location_basis="caller_confirmed_coordinates"
            )
            return selected
        if len(known) == 1:
            return dict(
                known[0],
                destination=destination,
                scope="city",
                location_basis="registered_city_default",
            )
        if len(known) > 1:
            return WeatherContext(
                status="needs_location",
                destination=destination,
                reason_code="ambiguous_location",
                reason="Multiple cached cities share this name",
                location_candidates=[candidate(c) for c in known],
                next_steps=["Choose a city with its country and coordinates."],
            )
        query = {"name": destination, "count": 5, "language": "en"}
        _, separator, qualifier = destination.replace("，", ",").partition(",")
        qualifier_country = country_for(qualifier.strip(), self.registry) if separator else None
        if qualifier_country:
            query["countryCode"] = qualifier_country["code"]
        source = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(query)
        try:
            payload = self.transport(source)
            results = payload.get("results", [])
            cities = [r for r in results if r.get("feature_code", "").startswith("PPL")]
            if len(results) == 1 and results[0].get("feature_code") == "PCLI":
                return WeatherContext(
                    status="needs_location",
                    destination=destination,
                    source=source,
                    reason_code="country_requires_city",
                    reason="The search resolved to a country, not a city weather location",
                    next_steps=["Name a city in that country; no national climate mean is used."],
                )
            if len(results) != 1 or len(cities) != 1:
                return WeatherContext(
                    status="needs_location",
                    destination=destination,
                    source=source,
                    reason_code="ambiguous_location" if results else "location_not_found",
                    reason="Choose a city; results are ambiguous, absent or outside city scope",
                    location_candidates=[
                        candidate(
                            dict(
                                r,
                                source=source,
                                retrieved_at=datetime.now(timezone.utc).isoformat(),
                                scope="city"
                                if r.get("feature_code", "").startswith("PPL")
                                else "non_city",
                            )
                        )
                        for r in results
                    ],
                    next_steps=[
                        "Provide city and country, or confirm a city coordinate candidate."
                    ],
                )
            resolved = dict(
                cities[0],
                destination=destination,
                scope="city",
                source=source,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                location_basis="provider_geocoded_city",
            )
            validate_city_location(resolved)
            return resolved
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return WeatherContext(
                destination=destination,
                source=source,
                reason_code="geocoding_unavailable",
                reason=f"Location lookup unavailable: {type(exc).__name__}",
                next_steps=[
                    "Retry online or provide a confirmed city, country, timezone and coordinates."
                ],
            )

    @staticmethod
    def scope_question(destination):
        return WeatherContext(
            status="needs_location",
            destination=destination,
            reason_code="non_city_scope",
            reason="City weather does not describe indoor, mountain or coastal microclimates",
            next_steps=[
                "Explicitly choose a city for regional reference, or use a local weather source."
            ],
        )
