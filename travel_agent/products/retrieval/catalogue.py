import csv
import json
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from .schemas import (
    Category,
    Garment,
    ProductRetrievalRequest,
    ProductRetrievalResponse,
    RawProduct,
    RecordIssue,
    SourceType,
)


class CatalogueError(ValueError):
    """File-level error; individual bad records are reported separately."""


ALIASES = {
    "shirt": Category.TOP,
    "t-shirt": Category.TOP,
    "top": Category.TOP,
    "trousers": Category.BOTTOM,
    "pants": Category.BOTTOM,
    "bottom": Category.BOTTOM,
    "shoes": Category.FOOTWEAR,
    "sneakers": Category.FOOTWEAR,
    "footwear": Category.FOOTWEAR,
    "jacket": Category.OUTERWEAR,
    "coat": Category.OUTERWEAR,
    "outerwear": Category.OUTERWEAR,
}


def normalize(product: RawProduct) -> Garment:
    category = ALIASES.get(product.raw_category.casefold(), Category.UNKNOWN)
    return Garment(**product.model_dump(), category=category)


class ProductProvider(Protocol):
    def retrieve(self, request: ProductRetrievalRequest) -> ProductRetrievalResponse: ...


class PreparedCatalogueProvider:
    def __init__(self, path):
        self.path = Path(path)
        self.products = []
        self.issues = []
        self._load()

    def _load(self):
        try:
            if self.path.suffix.lower() == ".json":
                records = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(records, list):
                    raise CatalogueError("JSON catalogue must be a list")
            elif self.path.suffix.lower() == ".csv":
                with self.path.open(encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    if not {"product_id", "title", "raw_category"}.issubset(
                        reader.fieldnames or []
                    ):
                        raise CatalogueError("CSV requires product_id, title, raw_category")
                    records = list(reader)
            else:
                raise CatalogueError("Supported formats: .json and .csv")
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
            raise CatalogueError(f"Cannot read catalogue {self.path}: {exc}") from exc
        seen = set()
        for index, record in enumerate(records, 1):
            pid = record.get("product_id") if isinstance(record, dict) else None
            try:
                if self.path.suffix.lower() == ".csv":
                    record = {k: v for k, v in record.items() if v != ""}
                    for key in ("price", "sizes", "evidence", "embedding"):
                        if key in record:
                            record[key] = json.loads(record[key])
                raw = RawProduct.model_validate(record)
                if raw.source_type != SourceType.PREPARED_CATALOGUE:
                    raise ValueError("prepared provider accepts PREPARED_CATALOGUE only")
                if raw.product_id in seen:
                    raise ValueError("duplicate product_id; first valid record wins")
                seen.add(raw.product_id)
                self.products.append(normalize(raw))
            except (ValidationError, ValueError, TypeError) as exc:
                self.issues.append(
                    RecordIssue(
                        record_index=index,
                        product_id=str(pid) if pid is not None else None,
                        reason=str(exc),
                    )
                )

    def retrieve(self, request):
        query = (
            request.query if request.query is not None else request.context.query or ""
        ).casefold()
        matches = []
        for g in self.products:
            if query and query not in (g.title + " " + g.description).casefold():
                continue
            if request.categories and g.category not in request.categories:
                continue
            if request.size and request.size.casefold() not in [s.casefold() for s in g.sizes]:
                continue
            if request.currency and (not g.price or g.price.currency != request.currency):
                continue
            if request.max_price is not None and (
                not g.price or g.price.amount > request.max_price
            ):
                continue
            matches.append(g)
        return ProductRetrievalResponse(
            products=matches[: request.limit],
            total_matches=len(matches),
            wardrobe_garments=request.wardrobe_garments,
            pending_wardrobe_uploads=request.wardrobe_uploads,
            issues=self.issues,
        )
