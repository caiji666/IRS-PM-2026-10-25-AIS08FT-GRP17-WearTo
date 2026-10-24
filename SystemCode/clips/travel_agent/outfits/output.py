"""File handoff helpers; original budget and native purchase prices stay distinct."""

import json
import os
import tempfile
from pathlib import Path


def recommendation_input(result, understanding):
    from travel_agent.recommendation.projection import project_trip_request

    if result.status != "ok":
        raise ValueError("Student 4 must be ready before preparing Student 5 input")
    data = understanding.get("data", understanding)
    request = project_trip_request(data["trip_request"])
    return {
        "action": "recommend",
        "request": request.model_dump(mode="json"),
        "student4": {k: result.data[k] for k in ("status", "source", "model_version", "outfits")},
    }


def save_artifact(value, target, source_report):
    """Register derived Production data without running a cleanup worker."""
    target = Path(target)

    def write():
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    if source_report.get("environment") != "production":
        write()
        return
    from travel_agent.products.notifications.lifecycle import (
        DeletedAccount,
        Registry,
        lifecycle_lock,
    )

    root = Path(os.environ["EBAY_MANAGED_ROOT"]).resolve()
    with lifecycle_lock(root):
        registry = Registry(root, root / ".cache/ebay-lifecycle/registry.sqlite3")
        try:
            if source_report.get("lifecycle_generation") != registry.generation():
                raise DeletedAccount("Source report invalidated; rerun product retrieval")
            registry.register(target.resolve(), "outfit_handoff")
            write()
        finally:
            registry.close()
