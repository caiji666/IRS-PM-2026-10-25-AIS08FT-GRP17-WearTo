"""Local/shared-agent adapter. Core M1-M6 contracts remain unchanged."""

import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import Field

from .schemas import Contract, ExpandedFashionContext, ProductRetrievalRequest, Trip


class ProductToolInput(Contract):
    understanding: dict | None = None
    request: dict | None = None
    upstream: dict[str, str] | None = None
    source: str = Field(default="photos20", pattern="^(photos20|prepared|ebay)$")
    search_mode: str = Field(default="hybrid", pattern="^(image|hybrid|keyword)$")
    reference_mode: str = Field(default="recompute", pattern="^(recompute|imported)$")
    k_per_category: int = Field(default=5, ge=1, le=100)
    wardrobe_images: list[str] = Field(default_factory=list)


@dataclass
class Settings:
    data_root: Path
    work_root: Path
    model_cache: Path
    credentials: Path | None = None
    offline: bool = True

    @classmethod
    def from_environment(cls):
        root = Path(os.environ.get("STUDENT3_DATA_ROOT", "artifacts/product-retrieval")).resolve()
        work = Path(
            os.environ.get("STUDENT3_WORK_ROOT", "artifacts/product-retrieval-runs")
        ).resolve()
        cache = Path(os.environ.get("STUDENT3_MODEL_CACHE", str(root / ".cache/huggingface")))
        secret = os.environ.get("STUDENT3_EBAY_CREDENTIALS")
        return cls(root, work, cache, Path(secret) if secret else None)

    def input_path(self, relative):
        path = (self.data_root / relative).resolve()
        path.relative_to(self.data_root.resolve())
        return path


def map_understanding(envelope):
    """Map Student 1 ModuleResult, preserving the full original in a sidecar."""
    if envelope.get("status") != "ok":
        raise ValueError("Student 1 must return status=ok before product retrieval")
    data = envelope.get("data", {})
    if data.get("ready_for_downstream") is not True:
        raise ValueError("Student 1 requirements are not ready for downstream use")
    trip = data.get("trip_request", {})
    text = data.get("expanded_fashion_context")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Student 1 expanded_fashion_context is required")
    notes = {
        k: data.get(k)
        for k in (
            "season_context",
            "weather_clothing_requirements",
            "activity_style_requirements",
            "limitations",
        )
    }
    context = ExpandedFashionContext(
        trip=Trip(
            destination=trip.get("destination"),
            start_date=trip.get("start_date"),
            end_date=trip.get("end_date"),
        ),
        query=text,
        activities=trip.get("activities", []),
        style_preferences=trip.get("style", []),
        weather_notes=json.dumps(notes),
    )
    # The trip budget is an outfit-level constraint, not a per-product price ceiling.
    return context, {
        "student1": envelope,
        "hard_constraints": data.get("hard_constraints", {}),
        "feasibility_status": "unverified",
    }


def prepare_input(spec, settings):
    metadata = {}
    if spec.upstream:
        from .upstream import adapt_files

        required = {"pt", "text", "context"}
        if not required <= spec.upstream.keys() or set(spec.upstream) - required - {"model_info"}:
            raise ValueError("upstream requires pt/text/context and optional model_info")
        paths = {k: settings.input_path(v) for k, v in spec.upstream.items()}
        imported, fallback, metadata = adapt_files(
            paths["pt"], paths["text"], paths["context"], paths.get("model_info")
        )
        request = imported if spec.reference_mode == "imported" else fallback
        if spec.reference_mode == "imported" and metadata["status"] != "declared_compatible":
            raise ValueError("Imported vectors require compatible declared model metadata")
        if spec.reference_mode == "recompute":
            for i, ref in enumerate(request.top_social_references):
                # Original cached downloads can be reused offline, in the exact PT row order.
                for folder in (
                    "data/team_input/reference_images",
                    "runs/ebay_sandbox/reference_images",
                ):
                    path = settings.input_path(f"{folder}/{i:03d}.jpg")
                    if path.is_file():
                        ref.image_path = str(path)
                        break
                if not ref.image_path:
                    metadata.setdefault("adapter_warnings", []).append(
                        f"{ref.reference_id}: no cached raster; description fallback only"
                    )
    elif spec.request:
        request = ProductRetrievalRequest.model_validate(spec.request)
    else:
        raise ValueError("Provide upstream files or a ProductRetrievalRequest")
    if spec.understanding:
        context, student1 = map_understanding(spec.understanding)
        request.context = context
        request.query = None
        request.context_embedding = None  # Re-encode the updated Student 1 context.
        metadata.update(student1)
    for ref in request.top_social_references:
        if ref.image_path:
            ref.image_path = str(settings.input_path(ref.image_path))
    return request, metadata


def shared_candidates(bundle, currency):
    """Project only representable items; retain every item in the full bundle."""
    digits = {
        "SGD": 2,
        "USD": 2,
        "EUR": 2,
        "GBP": 2,
        "JPY": 0,
        "KRW": 0,
        "CNY": 2,
        "MYR": 2,
        "THB": 2,
        "VND": 0,
        "IDR": 2,
    }
    candidates, blocked = [], []
    items = bundle["owned_items"] + [
        i for group in bundle["top_k_products"].values() for i in group
    ]
    for item in items:
        price = item.get("price")
        owned = item["owned"]
        code = currency if owned else (price or {}).get("currency")
        reason = None
        if item["category"] == "UNKNOWN":
            reason = "unknown category"
        elif code not in digits:
            reason = "unknown/unsupported currency"
        elif not owned and (
            not price or price["is_synthetic"] or not item["evidence"]["price_verified"]
        ):
            reason = "unverified/unknown purchase price; retained in bundle"
        if reason:
            blocked.append({"id": item["product_id"], "reason": reason})
            continue
        amount = Decimal("0") if owned else Decimal(str(price["amount"])) * 10 ** digits[code]
        if amount != amount.to_integral_value():
            blocked.append({"id": item["product_id"], "reason": "price exceeds currency precision"})
            continue
        candidates.append(
            {
                "id": item["product_id"],
                "role": item["category"].lower(),
                "origin": "owned" if owned else "new",
                "purchase_cost_minor": int(amount),
                "currency": code,
                "availability": "unknown",
                "image_ref": item.get("image_path"),
                "attributes": {
                    "title": item["title"],
                    "colors": ", ".join(item["colors"]),
                    "source": item["source_type"],
                },
            }
        )
    return candidates, blocked


def execute(spec, settings, encoder=None, client=None):
    """Business logic entry; heavy dependencies and network clients are lazy."""
    from .m5 import CachedEncoder, FashionCLIPEncoder, build_index
    from .m6 import RankedProductProvider
    from .phase2.service import combine, ingest, load_store

    request, metadata = prepare_input(spec, settings)
    encoder = encoder or CachedEncoder(
        FashionCLIPEncoder(settings.model_cache, offline=settings.offline),
        settings.work_root / "embeddings",
    )
    # Each runtime identity has an independent store: never relabel existing embeddings.
    import hashlib

    profile = hashlib.sha256(encoder.identity.encode()).hexdigest()[:16]
    wardrobe = settings.work_root / "wardrobe" / profile
    store = load_store(wardrobe / "items.json")
    issues = []
    if spec.wardrobe_images:
        store, issues = ingest(
            [settings.input_path(p) for p in spec.wardrobe_images],
            encoder,
            wardrobe / "items.json",
            wardrobe / "images",
        )
    if spec.source == "ebay":
        if not spec.upstream:
            raise ValueError("eBay pipeline requires upstream files")
        from .ebay.client import BrowseClient
        from .ebay.pipeline import run

        if client is None:
            if settings.credentials is None:
                raise ValueError(
                    "Configure STUDENT3_EBAY_CREDENTIALS locally; never supply secrets to the tool"
                )
            client = BrowseClient(json.loads(settings.credentials.read_text()), "EBAY_US")
        paths = {k: settings.input_path(v) for k, v in spec.upstream.items()}
        api = run(
            client,
            encoder,
            paths["pt"],
            paths["text"],
            paths["context"],
            settings.work_root / "ebay",
            paths.get("model_info"),
            spec.reference_mode,
            5,
            20,
            40,
            spec.k_per_category,
            wardrobe / "items.json",
            request_override=request.model_dump(mode="json") if spec.understanding else None,
            search_mode=spec.search_mode,
        )
        bundle = api["bundle"]
        metadata["api_run"] = {k: v for k, v in api.items() if k != "bundle"}
    else:
        if spec.source == "photos20":
            from .photo_dataset import photo_index

            normalized, index, _ = photo_index(settings.input_path("data/photos20"), encoder)
        else:
            from .m3 import process_catalogue

            normalized = process_catalogue(settings.input_path("data/catalogue_phase1.json"))
            index = build_index(normalized.products, encoder, settings.data_root)
        exclusions = metadata.get("hard_constraints", {}).get("excluded_items", [])
        if exclusions:
            index = index.model_copy(
                update={
                    "items": [
                        i
                        for i in index.items
                        if not any(
                            str(e).casefold() in i.product.title.casefold()
                            or str(e).casefold() == i.product.category.value.casefold()
                            for e in exclusions
                        )
                    ]
                }
            )
        ranked = RankedProductProvider(index, encoder, settings.data_root).retrieve_grouped(
            request, spec.k_per_category
        )
        bundle = combine(store, ranked, encoder.identity, encoder.dimension, issues).model_dump(
            mode="json"
        )
        metadata["normalization"] = normalized.model_dump(mode="json")
    currency = metadata.get("hard_constraints", {}).get("currency") or request.currency
    candidates, blocked = shared_candidates(bundle, currency)
    return {
        "contract_version": "student3.products.1",
        "bundle": bundle,
        "candidates": candidates,
        "handoff_blockers": blocked,
        "upstream_metadata": metadata,
        "request_used": request.model_dump(mode="json"),
        "model_identity": encoder.identity,
        "feasibility_status": "unverified",
        "source": spec.source,
        "limitations": [
            "Semantic retrieval only; outfit generation and final validation remain downstream.",
            "Prepared photographs and Sandbox listings are not live purchasing offers.",
            "All items remain in bundle; only verified priced items project to ProductCandidate.",
        ],
    }
