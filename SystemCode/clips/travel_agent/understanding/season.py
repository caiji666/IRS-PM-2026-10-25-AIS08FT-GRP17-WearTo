"""Calendar season for retrieval, separate from weather and user-provided facts."""

import json
import math
from datetime import date
from pathlib import Path

from travel_agent.schemas import SeasonContext

SOURCES = [
    "https://weather.metoffice.gov.uk/learn-about/met-office-for-schools/other-content/other-resources/our-seasons",
    "https://www.bom.gov.au/news-and-media/solstices-equinoxes-and-the-seasons",
]
NORTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}
SOUTH = {"winter": "summer", "spring": "autumn", "summer": "winter", "autumn": "spring"}


def derive_season(request, weather=None, confirmed_location=None):
    result = SeasonContext(sources=list(SOURCES))
    if not request.start_date or not request.end_date:
        result.reason = "Exact travel dates are required; no season is guessed"
        return result
    if weather is not None and weather.status == "needs_location":
        result.reason = "Confirm the destination before deriving a season"
        return result
    latitude = None
    if (
        weather is not None
        and weather.destination == request.destination
        and weather.latitude is not None
    ):
        latitude = weather.latitude
        result.location_basis = "weather_provider_location"
        if weather.source:
            result.sources.append(weather.source)
    elif confirmed_location is not None:
        if confirmed_location.get("destination") != request.destination:
            result.reason = "Confirmed coordinates are not bound to this destination"
            return result
        latitude = confirmed_location.get("latitude")
        result.location_basis = "caller_confirmed_coordinates"
    else:
        registry = json.loads(Path(__file__).with_name("season_locations.json").read_text())
        for record in registry["records"]:
            if (request.destination or "").casefold() in {
                name.casefold() for name in [record["name"], *record["aliases"]]
            }:
                latitude = record["latitude"]
                result.location_basis = "registered_city_default"
                result.sources.append(record["source"])
                break
    if latitude is None:
        result.reason = "No supported destination coordinates; provide confirmed location"
        return result
    if (
        type(latitude) not in (int, float)
        or not math.isfinite(latitude)
        or not -90 <= latitude <= 90
    ):
        result.reason = "Invalid latitude; provide verified numeric coordinates"
        return result
    result.latitude = latitude
    result.hemisphere = "north" if latitude >= 0 else "south"
    # Conservative project scope, not a universal climatic classification. Do
    # not force a four-season label onto tropical/monsoon destinations.
    if abs(latitude) <= 23.5:
        result.status = "unsupported"
        result.reason = "Four-season calendar labels are not applied within the tropical scope"
        return result
    cursor = date(request.start_date.year, request.start_date.month, 1)
    final_month = date(request.end_date.year, request.end_date.month, 1)
    months = []
    while cursor <= final_month and len(months) < 12:
        months.append(cursor.month)
        if cursor == final_month:
            break
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    for month in months:
        season = NORTH[month]
        if latitude < 0:
            season = SOUTH[season]
        if season not in result.seasons:
            result.seasons.append(season)
    if len(result.seasons) == 1:
        result.season = result.seasons[0]
        result.status = "derived"
    else:
        result.status = "multiple"
        result.reason = "Trip spans multiple calendar seasons; no single season is selected"
    return result
