# Hybrid eBay product retrieval

This local feature extends eBay candidate acquisition. It does not replace FashionCLIP reranking, alter shared schemas, or treat social photos as owned clothes. Image search and keyword search are separate Browse requests. Both use the same environment and marketplace; Sandbox listings remain test data.

## Algorithm

1. Safely adapt the upstream PT, TXT and JSON files. A mapped Student 1 request may override the travel context while preserving the original upstream context in metadata.
2. Plan up to two distinct short English queries per requested category using explicit clothing nouns and modifiers in the travel context, expanded text and social descriptions. Generic category nouns are fallbacks; brands, gender and materials are not inferred. Unrecognized/non-English descriptions currently fall back to generic English category terms.
3. Download up to five social images by default. Recompute their FashionCLIP embeddings unless compatible upstream model metadata is supplied. Submit images to image search and queries to keyword search.
4. Deduplicate by eBay item ID. Record `retrieval_sources`, `matched_reference_ids`, and `matched_queries`. Search query labels never establish the actual product category.
5. Process candidates round-robin across known API categories, fetch item details and images, normalize through M3 and encode actual product images with FashionCLIP. UNKNOWN stock remains UNKNOWN; no implicit FX conversion.
6. Rank with 0.5 context weight and 0.5 social weight for hybrid/keyword modes. Social similarity is the maximum over usable reference images. Without usable social references, normalize to context-only scoring. The legacy image-only Python mode retains 0.7/0.3 unless explicitly overridden.
7. For each category with fewer than K eligible ranked products, attempt its broad fallback once. Previously executed queries are not repeated. Reprocess new products and rerank. The default search budget is 17 logical searches (up to five images, eight primary queries and four broad fallbacks); HTTP retries, OAuth, item detail requests and image downloads are separate. User limits can prevent some queries from running.
8. Return each category's actual Top-K plus the saved owned-item pool. No duplicate padding. Processing is capped at 40 indexed products by default; unprocessed candidates and category shortages are reported. Filters can reduce results further.

HTTP failures are recorded as issues, separately from successful empty responses. OAuth failures stop the run. Budget/size filters and missing images can all reduce output. Sandbox may remain empty even with both search paths; this is not evidence of Production availability.

## Run from a fresh checkout

Run in the repository root after following `product-retrieval-runbook.md` for Python 3.11, the products extra, pinned model download, upstream files and local credentials. Never commit credentials, upstream private inputs, downloaded photos or model weights.

```bash
uv sync --locked --extra products
uv run python -m travel_agent.products.retrieval.ebay \
  --credentials .secrets/ebay-sandbox.json auth-check
uv run python -m travel_agent.products.retrieval.ebay \
  --credentials .secrets/ebay-sandbox.json run \
  --pt data/team_input/phase_2_output.pt \
  --text data/team_input/expanded_fashion_context.txt \
  --context data/team_input/context.json \
  --search-mode hybrid --max-references 5 --limit 20 \
  --max-products 40 --max-search-requests 17 --k-per-category 5 \
  --output-dir runs/ebay_hybrid_sandbox
```

Optional `--request path/to/mapped-request.json` accepts the existing ProductRetrievalRequest schema, not a raw Student 1 envelope. The shared LangChain product tool accepts the original Student 1 `understanding` envelope, `source: "ebay"`, upstream paths, and optional `search_mode: "hybrid"` (default). It performs the existing mapping automatically. Use `--search-mode image` to retain image-only acquisition; keyword mode still downloads social images for final similarity scoring.

`result.json` records raw responses, query plan, scoring configuration, source counts, issues, normalized image index and the wardrobe/product bundle. `reference_images/` contains social references; `product_images/` contains downloaded candidate images. Product scores are in `bundle.top_k_products`; the original API item and retrieval provenance are in `index.items[].product.provenance.original`. Exit 0 means usable indexed products; exit 3 means none, with the reason in the report; exit 2 indicates a fatal error. An exit-0 result can still have category shortages or unavailable social evidence.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
ITSA_REAL_MODEL_TEST=1 uv run pytest tests/product_retrieval/test_ebay.py -q
```

Mock transports validate nonempty hybrid retrieval and real-model image encoding independently of Sandbox inventory. Live Sandbox validation must be reported separately and never substituted with prepared products.

Sandbox may emit HTTP URLs on `i.ebayimg.sandbox.ebay.com`; only that exact CDN is upgraded to HTTPS and allowlisted in Sandbox. Production retains its original HTTPS-only CDN policy. Missing or inaccessible images are excluded, not replaced with synthetic pictures.
