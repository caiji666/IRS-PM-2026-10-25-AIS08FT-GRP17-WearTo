"""Run Understanding -> live Unsplash/FashionCLIP -> live eBay product retrieval.

Default input is the structured Seoul example, including explicitly synthetic weather
and draft KG opt-in. This does not invoke natural-language Qwen extraction or a live
weather forecast. Both image retrieval APIs and FashionCLIP inference are real.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("examples/understanding/seoul_winter.json")
    )
    parser.add_argument("--max-queries", type=int, default=2)
    parser.add_argument("--per-page", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-products", type=int, default=80)
    parser.add_argument("--max-references", type=int, default=10)
    parser.add_argument("--max-search-requests", type=int, default=25)
    parser.add_argument("--k-per-category", type=int, default=5)
    parser.add_argument(
        "--with-outfits",
        action="store_true",
        help="Run dynamic Student4 after writing the current Student3 handoff",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    os.environ["HF_HUB_CACHE"] = str(root / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["EBAY_MANAGED_ROOT"] = str(root)
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f%z")
    stages = root / "runs/live_pipeline" / timestamp
    stages.mkdir(parents=True, exist_ok=False)
    os.environ["STUDENT3_DATA_ROOT"] = str(stages)
    record = {
        "status": "running",
        "started_at": timestamp,
        "input": str(args.input),
        "stages_dir": str(stages),
    }
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".secrets/unsplash.env", override=True)
        if not os.getenv("UNSPLASH_ACCESS_KEY"):
            raise ValueError("Configure .secrets/unsplash.env before running")
        import torch

        from travel_agent.inspiration.handoff import save_inspiration_handoff
        from travel_agent.inspiration.pipeline import run_inspiration_pipeline
        from travel_agent.inspiration.service import prepare_inspiration_context
        from travel_agent.products.notifications.lifecycle import save_report
        from travel_agent.products.retrieval.ebay.client import BrowseClient
        from travel_agent.products.retrieval.ebay.handoff import write_handoff
        from travel_agent.products.retrieval.ebay.pipeline import run as retrieve
        from travel_agent.products.retrieval.integration import map_understanding
        from travel_agent.products.retrieval.m5 import CachedEncoder, FashionCLIPEncoder
        from travel_agent.products.retrieval.schemas import ProductRetrievalRequest
        from travel_agent.understanding.service import run as understand

        payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
        record["stage"] = "understanding"
        first = understand(payload)
        envelope = first.model_dump(mode="json")
        (stages / "understanding.json").write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2) + "\n"
        )
        if first.status == "error":
            record.update(status="failed", error_type="UnderstandingProviderError")
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "stage": "understanding",
                        "error_type": "UnderstandingProviderError",
                    }
                ),
                flush=True,
            )
            return 2
        if first.status != "ok" or first.data.get("ready_for_downstream") is not True:
            record.update(status="needs_input", questions=first.data.get("questions", []))
            print(
                json.dumps(
                    {
                        "status": "needs_input",
                        "stage": "understanding",
                        "questions": record["questions"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 4
        # Do not bypass upstream readiness or invent missing requirements.
        context, metadata = map_understanding(envelope)
        expanded, structured = prepare_inspiration_context(first.data)
        record["understanding_limitations"] = first.data.get("limitations", [])
        record["hard_constraints"] = metadata.get("hard_constraints", {})
        print("Student 1: ready; limitations preserved in pipeline_record.json", flush=True)
        record["stage"] = "inspiration"
        second = run_inspiration_pipeline(
            expanded,
            structured,
            top_n=args.top_n,
            max_queries=args.max_queries,
            per_page=args.per_page,
        )
        if second is None:
            raise RuntimeError("No usable Unsplash inspiration output")
        phase, evaluation = second
        paths = save_inspiration_handoff(phase, expanded, structured)["upstream"]
        upstream = {key: stages / path for key, path in paths.items()}
        torch.save(evaluation, upstream["pt"].parent / "evaluation_data.pt")
        record["upstream"] = {key: str(path) for key, path in upstream.items()}
        record["social_references"] = len(phase["top_images"])
        print(f"Student 2: {len(phase['top_images'])} fresh Unsplash references", flush=True)
        # Release the Stage 2 model before loading the product encoder.
        del phase, second
        record["stage"] = "products"
        encoder = CachedEncoder(
            FashionCLIPEncoder(root / ".cache/huggingface", offline=True),
            root / ".cache/embeddings",
        )
        client = BrowseClient(
            json.loads((root / ".secrets/ebay-production.json").read_text()), "EBAY_US"
        )
        if client.environment != "production":
            raise ValueError("Production credentials required")
        diagnostic = root / "runs/product_retrieval_diagnostics" / timestamp
        override = ProductRetrievalRequest(context=context).model_dump(mode="json")
        record["stage"] = "products"
        result = retrieve(
            client,
            encoder,
            upstream["pt"],
            upstream["text"],
            upstream["context"],
            diagnostic,
            model_info=upstream["model_info"],
            reference_mode="imported",
            max_references=args.max_references,
            limit=20,
            max_products=args.max_products,
            k=args.k_per_category,
            wardrobe_store=root / "runs/wardrobe/items.json",
            request_override=override,
            search_mode="hybrid",
            max_search_requests=args.max_search_requests,
            context_weight=0.5,
        )
        result["upstream_metadata"]["student1"] = envelope
        result["upstream_metadata"]["hard_constraints"] = metadata.get("hard_constraints", {})
        save_report(result, diagnostic / "run_report.json")
        target = root / "runs/product_retrieval" / timestamp
        handoff = write_handoff(result, target)
        record.update(status=result["status"], output=str(handoff), summary=result["summary"])
        if args.with_outfits:
            from travel_agent.outfits.service import run as score_outfits

            record["stage"] = "outfits"
            fourth = score_outfits(
                {
                    "product_result_path": str(handoff),
                    "understanding": envelope,
                    "ranking_request": result["request_used"],
                }
            )
            output4 = root / "runs/outfit_generation" / timestamp
            output4.mkdir(parents=True, exist_ok=False)
            path4 = output4 / "result.json"
            from travel_agent.outfits.output import recommendation_input, save_artifact

            save_artifact(fourth.model_dump(mode="json"), path4, result)
            if fourth.status == "ok":
                input5 = output4 / "student5_input.json"
                save_artifact(recommendation_input(fourth, envelope), input5, result)
            record["outfits"] = {
                "status": fourth.status,
                "output": str(path4),
                "summary": fourth.data.get("summary", {}),
                "reason": fourth.reason,
                "student5_input": str(input5) if fourth.status == "ok" else None,
            }
            print("Student 4: " + json.dumps(record["outfits"]), flush=True)
            if fourth.status != "ok":
                record["status"] = fourth.status
                return 4 if fourth.status == "needs_input" else 3
        print(
            json.dumps(
                {"status": record["status"], "output": str(handoff), **result["summary"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if result["status"] == "completed" else 3
    except Exception as exc:
        # Avoid echoing credentials or arbitrary HTTP bodies.
        record.update(status="failed", error_type=type(exc).__name__)
        print(
            json.dumps(
                {
                    "status": "failed",
                    "stage": record.get("stage", "setup"),
                    "error_type": type(exc).__name__,
                }
            ),
            flush=True,
        )
        return 2
    finally:
        (stages / "pipeline_record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
