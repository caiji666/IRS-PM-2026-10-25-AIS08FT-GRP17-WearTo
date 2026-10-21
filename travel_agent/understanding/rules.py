"""Deterministic rule evaluation with traceable predicates and unknown handling."""

import json
from pathlib import Path

from travel_agent.schemas import WeatherClothingRequirements

RULE_PATH = Path(__file__).with_name("weather_rules.json")


def reason_weather(weather):
    rules = json.loads(RULE_PATH.read_text())
    output = WeatherClothingRequirements()
    trace = []
    if weather.status != "ok":
        return weather, output, trace
    weather = weather.model_copy(deep=True)
    weather.weather_class = None
    lo, hi = weather.temp_min_celsius, weather.temp_max_celsius
    historical = weather.data_type == "historical_climatology"
    if historical:
        if weather.climate_reference is None:
            return weather, output, trace
        months = weather.climate_reference.monthly
        lo = min(m.mean_daily_min_celsius for m in months)
        hi = max(m.mean_daily_max_celsius for m in months)
    thresholds = rules["classification"]
    if lo is not None and lo < thresholds["cold_min_below"]:
        weather.weather_class = "COLD"
    elif lo is not None and lo < thresholds["cool_min_below"]:
        weather.weather_class = "COOL"
    elif hi is not None and hi >= thresholds["hot_max_at_least"]:
        weather.weather_class = "HOT"
    elif lo is not None and hi is not None:
        weather.weather_class = "MILD"
    for rule in rules["rules"]:
        if historical:
            # Only temperature-based planning suggestions. Past wet/snowfall frequency
            # is never fed to a forecast probability/boolean rule.
            if rule["field"] not in {"temp_min_celsius", "temp_max_celsius"}:
                continue
            value = lo if rule["field"] == "temp_min_celsius" else hi
        else:
            value = getattr(weather, rule["field"])
        if value is None:
            continue
        threshold = rule["threshold"]
        matches = {
            "lt": lambda: value < threshold,
            "gte": lambda: value >= threshold,
            "range": lambda: threshold[0] <= value < threshold[1],
            "eq": lambda: value == threshold,
        }[rule["operator"]]()
        if not matches:
            continue
        for role, items in rule["output"].items():
            current = getattr(output, role)
            current.extend(item for item in items if item not in current)
        record = {
            "rule_id": rule["id"],
            "field": rule["field"],
            "observed": value,
            "operator": rule["operator"],
            "threshold": threshold,
            "version": rules["version"],
            "review_status": rules["review_status"],
            "basis": rules["basis"],
            "strength": "recommendation",
        }
        if historical:
            record.update(
                data_type="historical_climatology",
                field="monthly_mean_daily_min_celsius"
                if rule["field"] == "temp_min_celsius"
                else "monthly_mean_daily_max_celsius",
                baseline_start_year=weather.climate_reference.baseline_start_year,
                baseline_end_year=weather.climate_reference.baseline_end_year,
                limitation="Planning suggestion from monthly means, not a trip-date forecast",
            )
        trace.append(record)
    return weather, output, trace
