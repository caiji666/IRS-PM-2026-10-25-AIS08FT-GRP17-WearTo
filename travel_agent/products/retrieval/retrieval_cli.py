"""Actual FashionCLIP CLI for M4–M6; python -m travel_agent.products.retrieval.retrieval_cli --help."""

import argparse
import json
import os
import sys
from pathlib import Path

from .m3 import process_catalogue
from .m4 import WardrobeAnalyzer
from .m4.adapter import wardrobe_index
from .m5 import CachedEncoder, FashionCLIPEncoder, ProductIndex, build_index
from .m6 import RankedProductProvider, RankingConfig
from .schemas import ProductRetrievalRequest, WardrobeUpload


def main():
    parser = argparse.ArgumentParser(description="Student 3 M4–M6 real FashionCLIP")
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--embedding-cache", type=Path, default=Path(".cache/embeddings"))
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--offline", action="store_true", help="require already downloaded model")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index")
    index_parser.add_argument("--catalogue", type=Path, default=Path("data/catalogue_phase1.json"))
    index_parser.add_argument(
        "--normalized", type=Path, help="optional M3 report instead of M2 catalogue"
    )
    index_parser.add_argument("--output", type=Path, required=True)
    wardrobe = commands.add_parser("wardrobe")
    wardrobe.add_argument(
        "--uploads", type=Path, required=True, help="JSON array of M1 WardrobeUpload objects"
    )
    wardrobe.add_argument("--output", type=Path, required=True)
    wardrobe.add_argument("--minimum-similarity", type=float, default=0.20)
    wardrobe.add_argument("--minimum-margin", type=float, default=0.03)
    retrieval = commands.add_parser("retrieve")
    retrieval.add_argument("--index", type=Path, required=True)
    retrieval.add_argument("--request", type=Path, default=Path("fixtures/request.json"))
    retrieval.add_argument("--config", type=Path, help="RankingConfig JSON")
    retrieval.add_argument("--output", type=Path, required=True)
    retrieval.add_argument(
        "--per-category", action="store_true", help="return independent Top-K groups"
    )
    retrieval.add_argument("--k-per-category", type=int, default=5)
    retrieval.add_argument("--legacy-output", type=Path, help="optional unchanged M1 response")
    demo = commands.add_parser("demo")
    demo.add_argument("--catalogue", type=Path, default=Path("data/catalogue_phase1.json"))
    demo.add_argument("--request", type=Path, default=Path("fixtures/phase1_request.json"))
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--k-per-category", type=int, default=5)
    args = parser.parse_args()
    try:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HOME", str(args.model_cache.resolve()))
        encoder = CachedEncoder(
            FashionCLIPEncoder(
                args.model_cache,
                offline=args.offline,
                device=args.device,
                batch_size=args.batch_size,
            ),
            args.embedding_cache,
        )
        if args.command == "wardrobe":
            records = json.loads(args.uploads.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError("uploads must be a JSON array")
            uploads = [WardrobeUpload.model_validate(x) for x in records]
            result = WardrobeAnalyzer(
                encoder, args.image_root, args.minimum_similarity, args.minimum_margin
            ).analyze(uploads)
        elif args.command == "index":
            if args.normalized:
                from .m3.models import ProcessingReport

                normalized = ProcessingReport.model_validate_json(
                    args.normalized.read_text(encoding="utf-8")
                )
            else:
                normalized = process_catalogue(args.catalogue)
            result = build_index(normalized.products, encoder, args.image_root)
            from .m5.models import IndexIssue

            result.issues.extend(
                IndexIssue(reason=f"{i.stage} record {i.record_index}: {i.reason}")
                for i in normalized.issues
            )
        else:
            request = ProductRetrievalRequest.model_validate_json(
                args.request.read_text(encoding="utf-8")
            )
            config = (
                RankingConfig.model_validate_json(args.config.read_text())
                if args.command == "retrieve" and args.config
                else None
            )
            if args.command == "demo":
                normalized = process_catalogue(args.catalogue)
                index = build_index(normalized.products, encoder, args.image_root)
                provider = RankedProductProvider(
                    index, encoder, args.image_root, processing_issues=normalized.issues
                )
                wardrobe_result = WardrobeAnalyzer(encoder, args.image_root).analyze(
                    request.wardrobe_uploads
                )
                wardrobe_embeddings = wardrobe_index(wardrobe_result, encoder)
                wardrobe_ranked = RankedProductProvider(
                    wardrobe_embeddings, encoder, args.image_root
                ).retrieve_ranked(request)
                ranked = provider.retrieve_ranked(request)
                grouped = provider.retrieve_grouped(request, args.k_per_category)
                legacy = provider.retrieve(request)
                legacy.wardrobe_garments.extend(x.garment for x in wardrobe_result.products)
                successful = {x.upload.upload_id for x in wardrobe_result.products}
                legacy.pending_wardrobe_uploads = [
                    u for u in request.wardrobe_uploads if u.upload_id not in successful
                ]
                result = {
                    "schema_version": "m456.2",
                    "normalization": normalized.model_dump(mode="json"),
                    "index": index.model_dump(mode="json"),
                    "wardrobe": wardrobe_result.model_dump(mode="json"),
                    "ranking": ranked.model_dump(mode="json"),
                    "grouped_ranking": grouped.model_dump(mode="json"),
                    "wardrobe_ranking": wardrobe_ranked.model_dump(mode="json"),
                    "m1_response": legacy.model_dump(mode="json"),
                    "cache": {"hits": encoder.hits, "misses": encoder.misses},
                }
            else:
                index = ProductIndex.model_validate_json(args.index.read_text(encoding="utf-8"))
                provider = RankedProductProvider(index, encoder, args.image_root, config)
                result = (
                    provider.retrieve_grouped(request, args.k_per_category)
                    if args.per_category
                    else provider.retrieve_ranked(request)
                )
                if args.legacy_output:
                    args.legacy_output.parent.mkdir(parents=True, exist_ok=True)
                    args.legacy_output.write_text(
                        provider.retrieve(request).model_dump_json(indent=2) + "\n"
                    )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(result, indent=2, ensure_ascii=False)
            if isinstance(result, dict)
            else result.model_dump_json(indent=2)
        )
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved {args.output}; embedding cache hits={encoder.hits}, misses={encoder.misses}")
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
