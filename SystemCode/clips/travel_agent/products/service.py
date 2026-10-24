"""Student 3 adapter to the preserved product retrieval package."""

from pydantic import ValidationError

from travel_agent.schemas import ModuleResult, ProductCandidate


def run(payload: dict, *, settings=None, encoder=None, client=None) -> ModuleResult:
    """Validate explicit inputs, then lazily invoke the local retrieval pipeline."""
    if not payload:
        return ModuleResult(
            status="needs_input", reason="Provide upstream PT/TXT/JSON files or request"
        )
    try:
        from travel_agent.products.retrieval.integration import ProductToolInput, Settings, execute

        spec = ProductToolInput.model_validate(payload)
        if not spec.upstream and not spec.request:
            return ModuleResult(status="needs_input", reason="Provide upstream files or request")
        data = execute(spec, settings or Settings.from_environment(), encoder, client)
        data["candidates"] = [
            ProductCandidate.model_validate(x).model_dump(mode="json") for x in data["candidates"]
        ]
        count = sum(len(x) for x in data["bundle"]["top_k_products"].values())
        return ModuleResult(
            status="ok" if count else "no_candidates",
            data=data,
            reason=None if count else "No candidate products; owned pool remains available",
        )
    except (ValidationError, ValueError, OSError, ImportError, RuntimeError) as exc:
        return ModuleResult(status="error", reason=f"{type(exc).__name__}: {exc}")
