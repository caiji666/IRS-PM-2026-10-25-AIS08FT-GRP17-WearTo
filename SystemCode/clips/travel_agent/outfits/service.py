"""Student 4 public entry: dynamic Student 3 pools -> all scored outfit combinations."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.schemas import ModuleResult


class OutfitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    products: dict | None = None
    product_result_path: str | None = None
    understanding: dict | None = None
    ranking_request: dict | None = None
    base_dir: str = "."
    batch_size: int = Field(default=256, ge=1, le=2048)
    max_combinations: int = Field(default=200000, ge=1, le=1000000)


def run(payload: dict, *, model=None, manifest=None) -> ModuleResult:
    """Never reads a fixed sample. Model overrides are Python test dependencies, not JSON input."""
    try:
        spec = OutfitInput.model_validate(payload)
        if (spec.products is None) == (spec.product_result_path is None):
            raise ValueError("Provide exactly one of products or product_result_path")
        base = Path(spec.base_dir).resolve()
        if spec.product_result_path:
            path = Path(spec.product_result_path).expanduser().resolve()
            report = json.loads(path.read_text())
            base = path.parent
        else:
            report = spec.products
        if "bundle" not in report:
            raise ValueError("Student 3 products must contain a Phase 2 bundle")
        from .engine import load_model, score_outfits

        if model is None:
            model, manifest = load_model()
        elif manifest is None:
            raise ValueError("An injected test model requires a matching test manifest")
        result = score_outfits(
            report,
            spec.understanding,
            spec.ranking_request,
            base,
            model=model,
            manifest=manifest,
            batch_size=spec.batch_size,
            max_combinations=spec.max_combinations,
        )
        return ModuleResult(
            status=result["status"],
            data=result,
            reason=None
            if result["outfits"]
            else "No complete Student5-ready outfit; inspect scored combinations and issues",
        )
    except (ValueError, KeyError, OSError, ImportError) as exc:
        return ModuleResult(status="needs_input", reason=f"{type(exc).__name__}: {exc}")
    except RuntimeError as exc:
        return ModuleResult(status="error", reason=f"Outfit model inference failed: {exc}")
