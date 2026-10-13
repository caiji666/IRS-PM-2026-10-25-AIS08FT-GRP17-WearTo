# Travel Style Agent · Group 17

A shared LangChain agent project for the Intelligent Travel Style & Shopping Assistant. The Student 1 understanding module implements proposal-style request validation, patches, weather retrieval/rules, qualified KG reasoning and context aggregation, with optional local Qwen3-0.6B extraction. See [the Student 1 runbook](docs/student1-runbook.md) for runnable demos and required human review. The Student 2 Inspiration module implements FashionCLIP-based context and image embeddings, Unsplash fashion reference retrieval, deduplication, cosine similarity reranking, and Top-N selection, with a file-based handoff to Student 3. See [the Inspiration runbook](docs/inspiration-runbook.md). The Student 3 Product Retrieval module integrates the existing M1–M6 and wardrobe pipeline for product retrieval and ranking. See [the product retrieval runbook](docs/product-retrieval-runbook.md). The Student 5 recommendation module in `travel_agent/recommendation/` implements hard constraints, final Top-3 ranking, the user decision loop and outfit preview; see [Student 5 recommendation](#student-5-recommendation). Outfit scoring and image generation remain under development. Full live-agent orchestration across all modules has not yet been verified.

## Quick start

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/caiji666/travel-style-agent.git
cd travel-style-agent
uv sync --locked
uv run python -m travel_agent.demo
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Use Python 3.11 across the team. Commit both `pyproject.toml` and `uv.lock`. Add dependencies with `uv add` and include both files in the same pull request. Do not edit the lockfile manually.

The default demo uses explicit synthetic weather and draft KG knowledge, with no model or API calls. It demonstrates the implemented understanding pipeline, not model accuracy or complete outfit recommendations. The Inspiration module is implemented and requires its optional dependencies and Unsplash API credentials for live retrieval; its tool returns `needs_input` for the empty demo payload. The product retrieval requires explicit inputs and returns `needs_input` for the empty demo payload. The recommendation tool (Student 5) returns `error` because its payload requires an explicit `action`. The outfit tool (Student 4) returns `not_implemented`.

## English-first understanding workflow

Activity KG v0.2.0 now returns alternative aesthetics and independent dressing needs, covering all 11 Place KG activity IDs. See [activity style expansion](docs/activity-style-expansion.md) for source scope, draft examples and the downstream output contract. No new item mappings are part of this expansion.

English is the primary evaluated input language. Local Qwen3-0.6B proposes structured facts; a constrained English source-evidence layer checks explicit dates, money, activities, preferences, exclusions and edits. Raw model output and augmented results are retained separately. This is a supervised prototype, not a model-only accuracy claim. Relative/yearless dates, ambiguous currencies, unsupported edits and invalid inputs need clarification.

```bash
uv sync --locked --extra qwen
HF_HOME="$PWD/artifacts/huggingface" uv run --locked --extra qwen \
  python -m travel_agent.understanding.demo \
  --message "Seoul, 2026-12-20 to 2026-12-27. Sightseeing and dinner. Budget 500 SGD. I prefer minimalist style. Dinner requires smart casual dress code." \
  --output artifacts/extraction-preview.json
```

The first run returns a proposal for human review. Review `proposed_request` / `proposed_action` before explicitly accepting extraction in a direct CLI/service call. Do not enable automatic acceptance through the agent tool. Weather outside forecast coverage remains unavailable; no forecast is invented. Use the offline demo for a complete synthetic-weather walkthrough.

Current English diagnostics and preserved multilingual history are in [evaluation](evaluation/README.md). Chinese aliases inside copied KG snapshots are source provenance; public explanations use English labels. All knowledge/rule approvals remain pending. Optional Neo4j and a live multi-tool recommendation agent are not verified.

## Student 5 recommendation

`travel_agent.recommendation.service.run(payload)` is the Student 5 entry point (proposal Stages 10, 12, 13 and 14). The implementation lives in `travel_agent/recommendation/student5/` and the tests in `tests/recommendation/`.

Every payload names its operation with an explicit top-level `action`. A missing or unknown `action` returns `error`. The `action` is the service operation; it is distinct from the Stage 1 user intent, so a `patch` keeps its own `intent: "MODIFY_REQUIREMENTS"`.

| `action` | Required fields | Result |
|---|---|---|
| `recommend` | `request`, `student4` (optional `beta`, `session_id`, `state`) | Stages 10 → 12: Top-3 `RecommendationSet` and state (`ok`), or `no_candidates` with a structured conflict. |
| `modify_requirements` | `state`, `patch` | Updated state, changed fields and a selective `rerun_plan`. |
| `confirm` | `state`, latest `recommendation_set`, exact `outfit_id` | Confirmed state and `confirmed_outfit`. |
| `explain` | `state`, latest `recommendation_set`, exact `outfit_id` | Data-only explanation; the state is not changed. |
| `preview` | `confirmed_outfit`, `preview_mode` (optional `preview_provider: "mock"`) | Preview with `preview_source` `mock` or `product_card_fallback`. |

- **Student 4 input:** `student4` is Student 4's output, whose outfits already carry `compatibility_score`. Student 5 converts it with its `student4_adapter`; it does not compute CompatibilityScore itself. The Student 4 output format and this adapter remain PROVISIONAL until the real Student 4 hand-off.
- **Request projection:** `travel_agent.recommendation.projection.project_trip_request` builds the Student 5 request from Student 1's validated `trip_request`. It takes only eight shared fields: `destination`, `start_date`, `end_date`, `activities`, `budget` (major units), `currency`, `style` and `excluded_items`.
  - Student 1 extension fields (`owned_item_ids`, `places`, `dress_codes`, `enter_interior`, `gender`, `cold_sensitive`, `price_preference`) are not forwarded.
  - Missing required fields are reported by name, never invented.
  - `user_size` is not set because its source is still to be agreed; with `user_size=None` the size constraint is not enabled.
- **Stage 10:** hard constraints. Only purchase items count toward the budget, availability (a real boolean) is checked for purchase items, and size only when the size data is reliable. Constraints are never relaxed.
- **Stage 12:** final ranking with `FinalOutfitScore = beta * CompatibilityScore + (1 - beta) * OutfitProductRelevance` (`beta` is configurable) and stable `outfit_id`s.
- **Stage 14:** the optional real image provider is never created from a tool payload.
- **Import safety:** importing the service reads no `.env`, opens no network connection and loads no model.

## Five-module integration test

`tests/test_full_pipeline_integration.py` chains the real Student 1–5 entry points. The complete five-module pipeline has **not** been shown to pass end to end.

- **Student 1 → Student 2:** Student 2 receives Student 1's `ModuleResult.data` as `understanding`.
- **Student 2 → Student 3:** Student 2's `product_handoff` (upstream files, `reference_mode=imported`) is passed unchanged to Student 3.
- **Student 2 live runs:** need the `inspiration` extra, the FashionCLIP weights, `UNSPLASH_ACCESS_KEY` and network access.
- **Student 4:** `outfits.service.run` is still a scaffold, and is the current structural blocker.
- **Student 5:** reached through `travel_agent.recommendation.service.run` (`action: "recommend"`) with the minimal eight-field request projection. Real data cannot reach it until Student 4 returns real output.
- **Default tests:** offline. They download no model, call no external API and do not read `.env`. Contract and boundary checks run; the full-pipeline tests (`full_pipeline_live`) are skipped with the exact blocking stage until every real module runs. Model/API stages run only with `RUN_FULL_PIPELINE_TESTS=1`.

See [full-pipeline integration](docs/full_pipeline_integration.md) for the contracts, prerequisites, run commands and current gaps.

## Team responsibilities

- Student 1: `understanding/` — user input, structured requirements, weather reasoning, and knowledge graph reasoning.
- Student 2: `inspiration/` — context embeddings, social/web reference retrieval, and ranking.
- Student 3: `products/` — product retrieval, embeddings, and Top-K ranking.
- Student 4: `outfits/` and `training/` — Polyvore data, offline training, outfit generation, and compatibility scoring.
- Student 5: `recommendation/` — hard constraints, final ranking, user refinement, and previews.

Changes to shared schemas, `agent.py`, dependencies, and CI require review from affected team members. Ownership of main agent integration, the UI, deployment, and system-wide evaluation remains to be assigned by the team.

## LangChain integration

Implement module business logic in `service.py` and expose LangChain tools in `tools.py`. Create the main agent and register its tools centrally in `agent.py`. Each contributor develops a module of the shared agent system; a separate agent per contributor is not required.

`build_agent(model)` accepts a configured chat model instance or a LangChain model identifier. Install the appropriate provider package and configure credentials before using a live model. For example, when using the OpenAI integration:

```bash
uv sync --locked --extra openai
cp .env.example .env
# Set AGENT_MODEL (provider:model format) and the required API key in your local .env.
```

```python
import os
from dotenv import load_dotenv
from travel_agent.agent import build_agent

load_dotenv()
agent = build_agent(os.environ["AGENT_MODEL"])
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Help me plan outfits for my trip."}]},
    config={"recursion_limit": 12},
)
```

This example demonstrates integration only. Real recommendations require completed tool implementations. Persistent sessions are not configured yet; a shared checkpointer and session/thread ID convention will be added later. Budget calculations and final feasibility checks must be implemented in deterministic code. Prompts cannot replace constraint validation.

## Collaboration

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [interface contracts](docs/contracts.md).

Follow this workflow: main → feature branch → pull request → CI → peer review → squash merge. Keep full datasets, user-uploaded images, credentials, and model weights out of Git. Commit acquisition instructions, version information, and checksums instead.

English is the project language for documentation, code comments, examples, issues, and pull requests.

## Fashion Inspiration Retrieval and Ranking

The Inspiration module uses **FashionCLIP** to generate context and image embeddings, retrieve fashion references through the Unsplash API, and rank them using cosine similarity. It supports structured query generation, image deduplication, Top-N selection, and a file-based handoff to Student 3's product retrieval pipeline.

The module integrates with Student 1's Understanding output through the LangChain `search_references` tool.

**Setup:**

```bash
uv sync --locked --extra inspiration
```

Configure `UNSPLASH_ACCESS_KEY` in your local `.env` file.

**Standalone execution:**

```bash
uv run --no-sync python -m travel_agent.inspiration.pipeline
```

The standalone pipeline uses local context fixtures for testing. Full live-agent orchestration remains pending team integration.

For architecture, dependencies, integration contracts, and testing instructions, see the [Inspiration Runbook](docs/inspiration-runbook.md).

## Product retrieval and wardrobe commands

For a fresh checkout, follow the [product retrieval runbook](docs/product-retrieval-runbook.md). It covers the products extra, repository-local configuration, included offline demos, FashionCLIP download, personal wardrobe import, upstream PT/TXT/JSON preparation, optional historical photos, Sandbox credentials and output handoff. Private input files, complete datasets, user images, model weights and credentials are not included in Git. The runbook includes the clone command.

For the resource-complete current local checkout, see [independent local validation and commands](docs/local-product-retrieval-validation.md). Inputs, photographs, model files and private Sandbox configuration now reside within this checkout; the old standalone project is not required. These excluded resources remain local and are not supplied by a Git clone.



