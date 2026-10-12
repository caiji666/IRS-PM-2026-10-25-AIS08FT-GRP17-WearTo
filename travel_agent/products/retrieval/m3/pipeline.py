import copy
import re
from collections.abc import Mapping
from typing import Iterable

import pandas as pd
from pydantic import ValidationError

from ..catalogue import PreparedCatalogueProvider
from ..schemas import Availability, Category, RawProduct, SourceType
from .config import CATEGORY_ALIASES, COLOR_ALIASES, SIZE_ALIASES, STOCK_ALIASES
from .models import NormalizedProduct, ProcessingIssue, ProcessingReport, Provenance


def text(value):
    return re.sub(r"\s+", " ", value.strip()) if isinstance(value, str) else value


def category(value):
    token = text(value).casefold()
    for name, aliases in CATEGORY_ALIASES.items():
        if token in aliases:
            return Category(name)
    return Category.UNKNOWN


def labels(value, aliases, upper=False):
    if value is None:
        return []
    if isinstance(value, str):
        value = re.split(r"[,;/|]", value)
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError("attribute labels must be strings or a list of strings")
    result = []
    for item in value:
        item = text(item)
        if not item:
            continue
        normalized = aliases.get(item.casefold(), item.upper() if upper else item.casefold())
        if normalized not in result:
            result.append(normalized)
    return result


def process_records(records: Iterable, source="direct-input", record_indices=None):
    rows = list(records)
    indices = list(record_indices) if record_indices is not None else list(range(1, len(rows) + 1))
    if len(indices) != len(rows):
        raise ValueError("record_indices length mismatch")
    # Object-valued cells avoid Pandas coercing missing prices/IDs into fabricated values.
    frame = pd.DataFrame({"record": pd.Series(rows, dtype=object), "record_index": indices})
    report = ProcessingReport(input_count=len(rows))
    seen = set()
    for original, index in frame.itertuples(index=False, name=None):
        try:
            if not isinstance(original, Mapping):
                raise ValueError("record must be an object")
            data = copy.deepcopy(dict(original))
            data.pop("category", None)  # M2 category is recomputed from raw_category.
            colors = labels(data.pop("colors", None), COLOR_ALIASES)
            for field in ("product_id", "title", "raw_category", "description", "image_path"):
                if field in data:
                    data[field] = text(data[field])
            if not data.get("raw_category"):
                data["raw_category"] = "UNKNOWN"
            if data.get("description") is None:
                data["description"] = ""
            if data.get("image_path") == "":
                data["image_path"] = None
            data["sizes"] = labels(data.get("sizes"), SIZE_ALIASES, upper=True)
            if data.get("price") is not None:
                price = data["price"]
                if not isinstance(price, dict):
                    raise ValueError("price must contain amount and explicit currency")
                if isinstance(price.get("amount"), bool):
                    raise ValueError("boolean price amount is invalid")
                if isinstance(price.get("currency"), str):
                    price["currency"] = text(price["currency"]).upper()
            stock = data.get("availability")
            if stock is None or stock == "":
                data["availability"] = "UNKNOWN"
            elif isinstance(stock, str):
                data["availability"] = STOCK_ALIASES.get(
                    text(stock).casefold(), text(stock).upper()
                )
            raw = RawProduct.model_validate(data)
            notes = []
            if raw.availability.value != "UNKNOWN" and not raw.evidence.availability_verified:
                notes.append("Unverified stock downgraded to UNKNOWN.")
                raw = raw.model_copy(update={"availability": Availability.UNKNOWN})
            if category(raw.raw_category) == Category.UNKNOWN:
                notes.append("Category unresolved; retained UNKNOWN.")
            payload = raw.model_dump(mode="json")
            normalized = NormalizedProduct.model_validate(
                {
                    **payload,
                    "category": category(raw.raw_category),
                    "colors": colors,
                    "provenance": Provenance(
                        source=source, record_index=index, original=dict(original)
                    ),
                    "image_embedding_candidate": bool(
                        raw.image_path
                        and raw.evidence.image_verified
                        and raw.source_type != SourceType.PREPARED_CATALOGUE
                    ),
                }
            )
            key = (normalized.source_type, normalized.product_id)
            if key in seen:
                raise ValueError("duplicate source_type/product_id; first valid record wins")
            seen.add(key)
            report.products.append(normalized)
            report.issues.extend(ProcessingIssue(record_index=index, reason=n) for n in notes)
        except (ValidationError, ValueError, TypeError) as exc:
            report.issues.append(ProcessingIssue(record_index=index, reason=str(exc)))
    return report


def process_catalogue(path):
    """Load through the existing M2 provider without mutating its products or API."""
    provider = PreparedCatalogueProvider(path)
    rejected = {issue.record_index for issue in provider.issues}
    total = len(provider.products) + len(provider.issues)
    indices = [i for i in range(1, total + 1) if i not in rejected]
    records = []
    for garment in provider.products:
        record = garment.model_dump(mode="json")
        record.pop("category")
        records.append(record)
    report = process_records(records, source=str(provider.path.resolve()), record_indices=indices)
    report.input_count = total
    report.issues = [
        ProcessingIssue(record_index=i.record_index, stage="M2", reason=i.reason)
        for i in provider.issues
    ] + report.issues
    return report
