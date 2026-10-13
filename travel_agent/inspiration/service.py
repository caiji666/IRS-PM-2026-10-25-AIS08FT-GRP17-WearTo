"""Student 2: retrieve and rank fashion references."""

from travel_agent.schemas import ModuleResult
from .handoff import save_inspiration_handoff

import logging
from .pipeline import run_inspiration_pipeline
logger = logging.getLogger(__name__)

def prepare_inspiration_context(data: dict) -> tuple[str, dict]:
    """Convert Student 1 output into Student 2 pipeline inputs."""

    trip_request = data["trip_request"]

    expanded_context = data["expanded_fashion_context"]

    structured_context = {
        "destination": trip_request["destination"],
        "season": data["season"],
        "weather_clothing_requirements": data["weather_clothing_requirements"],
        "activity_style_requirements": data["activity_style_requirements"],
        "user_style": trip_request.get("style", []),
    }

    return expanded_context, structured_context


def run(payload: dict) -> ModuleResult:
    """Execute Phase 2 inspiration retrieval using Phase 1 output."""

    if not isinstance(payload, dict):
        return ModuleResult(
            status="needs_input",
            reason="Phase 1 output must be a dictionary."
        )

    # Phase 1's ModuleResult.data
    data = payload.get("understanding", payload)

    if not isinstance(data, dict):
        return ModuleResult(
            status="needs_input",
            reason="Understanding data must be a dictionary."
        )

    if data.get("ready_for_downstream") is not True:
        return ModuleResult(
            status="needs_input",
            reason="Phase 1 context is not ready for downstream processing."
        )

    try:
        expanded_context, structured_context = prepare_inspiration_context(data)

        if not isinstance(expanded_context, str) or not expanded_context.strip():
            return ModuleResult(
                status="needs_input",
                reason="ExpandedFashionContext is missing or empty."
            )

        if not structured_context.get("destination"):
            return ModuleResult(
                status="needs_input",
                reason="Trip destination is required."
            )

        top_n = payload.get("top_n", 10)

        if type(top_n) is not int or top_n <= 0:
            return ModuleResult(
                status="needs_input",
                reason="top_n must be a positive integer."
            )

        logger.info("Starting Phase 2 Inspiration Service")

        result = run_inspiration_pipeline(
            expanded_context=expanded_context,
            structured_context=structured_context,
            top_n=top_n
        )

        if result is None:
            return ModuleResult(
                status="no_candidates",
                reason="No usable fashion inspiration images were returned."
            )

        phase2_output, evaluation_data = result
        # Prepare Student 3 file-based handoff
        logger.info("Preparing Student 3 handoff")

        handoff = save_inspiration_handoff(
            phase2_output=phase2_output,
            expanded_context=expanded_context,
            structured_context=structured_context
        )

        logger.info("Student 3 handoff prepared successfully")

        return ModuleResult(
            status="ok",
            data={
                "context_embedding": phase2_output["context_embedding"].cpu().tolist(),
                "top_images": phase2_output["top_images"],
                "top_image_embeddings": (
                    phase2_output["top_image_embeddings"].cpu().tolist()
                ),
                "product_handoff": handoff
            }
        )

    except (KeyError, TypeError, ValueError) as error:
        logger.warning("Invalid Phase 1 context: %s", error)

        return ModuleResult(
            status="needs_input",
            reason="Phase 1 output is missing or contains invalid context fields."
        )

    except Exception:
        logger.exception("Phase 2 inspiration pipeline failed")

        return ModuleResult(
            status="error",
            reason="Inspiration retrieval or ranking failed."
        )
