from langchain.tools import tool

from travel_agent.products.service import run


@tool
def search_products(payload: dict) -> dict:
    """Student 3: retrieve and rank product candidates.

    Payload requires either `upstream` (relative pt/text/context paths, optional
    model_info) or `request` (a ProductRetrievalRequest object). Optional
    `understanding` is the ready Student 1 ModuleResult. Paths resolve under
    STUDENT3_DATA_ROOT. Select `source` as photos20, prepared or ebay;
    `reference_mode` as recompute (default) or strictly validated imported;
    `k_per_category` defaults to 5; `wardrobe_images` contains relative image paths.
    Credentials are local environment configuration and must never enter payload.
    Returns category Top-K, owned pool, provenance and explicit handoff blockers.
    Retrieval does not verify outfit budget, stock or final feasibility.
    """
    return run(payload).model_dump(mode="json")
