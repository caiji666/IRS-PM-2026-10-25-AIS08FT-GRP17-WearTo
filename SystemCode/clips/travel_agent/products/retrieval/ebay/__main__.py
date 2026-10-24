import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .client import BrowseClient


def main():
    p = argparse.ArgumentParser(
        description="eBay Browse image retrieval: explicit sandbox/production credentials"
    )
    p.add_argument("--credentials", type=Path, default=Path(".secrets/ebay-sandbox.json"))
    p.add_argument("--marketplace", default="EBAY_US")
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("auth-check")
    subs.add_parser("configure")
    r = subs.add_parser("run")
    r.add_argument("--search-mode", choices=["image", "hybrid", "keyword"], default="hybrid")
    r.add_argument("--max-search-requests", type=int, default=17)
    r.add_argument("--context-weight", type=float, default=0.5)
    r.add_argument(
        "--request", type=Path, help="Mapped ProductRetrievalRequest JSON from Student 1"
    )
    r.add_argument("--pt", type=Path, required=True)
    r.add_argument("--text", type=Path, required=True)
    r.add_argument("--context", type=Path, required=True)
    r.add_argument("--model-info", type=Path)
    r.add_argument("--reference-mode", choices=["recompute", "imported"], default="recompute")
    r.add_argument("--max-references", type=int, default=5)
    r.add_argument("--limit", type=int, default=20)
    r.add_argument("--max-products", type=int, default=40)
    r.add_argument("--k-per-category", type=int, default=5)
    r.add_argument("--category-ids")
    r.add_argument("--wardrobe-store", type=Path, default=Path("runs/wardrobe/items.json"))
    r.add_argument("--model-cache", type=Path, default=Path(".cache/huggingface"))
    r.add_argument("--embedding-cache", type=Path, default=Path(".cache/embeddings"))
    r.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/product_retrieval"),
        help="Parent for timestamped portable downstream output directories",
    )
    a = p.parse_args()
    try:
        if a.command == "configure":
            from .credentials import configure

            configure(a.credentials)
            print("Credentials saved locally; secret not displayed.")
            return 0
        client = BrowseClient(json.loads(a.credentials.read_text()), a.marketplace)
        if a.command == "auth-check":
            client.authenticate()
            print(f"eBay {client.environment} OAuth succeeded; token not displayed or persisted.")
            return 0
        from ..m5 import CachedEncoder, FashionCLIPEncoder
        from .pipeline import run

        timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f%z")
        handoff_dir = a.output_dir / timestamp
        diagnostics_dir = Path("runs/product_retrieval_diagnostics") / timestamp
        encoder = CachedEncoder(FashionCLIPEncoder(a.model_cache, offline=True), a.embedding_cache)
        result = run(
            client,
            encoder,
            a.pt,
            a.text,
            a.context,
            diagnostics_dir,
            a.model_info,
            a.reference_mode,
            a.max_references,
            a.limit,
            a.max_products,
            a.k_per_category,
            a.wardrobe_store,
            a.category_ids,
            request_override=json.loads(a.request.read_text()) if a.request else None,
            search_mode=a.search_mode,
            max_search_requests=a.max_search_requests,
            context_weight=a.context_weight,
        )
        from ...notifications.lifecycle import save_report

        save_report(result, diagnostics_dir / "run_report.json")
        from .handoff import write_handoff

        handoff = write_handoff(result, handoff_dir)
        print(f"Downstream output: {handoff}")
        print(json.dumps({"status": result["status"], **result["summary"]}, ensure_ascii=False))
        return 0 if result["status"] == "completed" else 3
    except Exception as exc:
        # HTTP client already omits credentials, tokens, headers and request bodies from errors.
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
