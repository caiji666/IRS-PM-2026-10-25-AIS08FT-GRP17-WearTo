# Travel Style Agent · Group 17

A shared LangChain agent project for the Intelligent Travel Style & Shopping Assistant. The Student 1 understanding module implements proposal-style request validation, patches, weather retrieval/rules, qualified KG reasoning and context aggregation, with optional local Qwen3-0.6B extraction. Student 3 product retrieval now wraps the existing M1–M6 and wardrobe pipeline; social retrieval, MLP scoring and image generation remain scaffolds. See [the product retrieval runbook](docs/product-retrieval-runbook.md). See [the Student 1 runbook](docs/student1-runbook.md) for runnable demos and required human review.

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

The default demo uses explicit synthetic weather and draft KG knowledge, with no model or API calls. It demonstrates the implemented understanding pipeline, not model accuracy or complete outfit recommendations. The social, outfit and recommendation tools return `not_implemented`; product retrieval requires explicit inputs and returns `needs_input` for the empty demo payload.

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

## Product retrieval and wardrobe commands

For a fresh checkout, follow the [product retrieval runbook](docs/product-retrieval-runbook.md). It covers the products extra, repository-local configuration, included offline demos, FashionCLIP download, personal wardrobe import, upstream PT/TXT/JSON preparation, optional historical photos, Sandbox credentials and output handoff. Private input files, complete datasets, user images, model weights and credentials are not included in Git. Use the `feat/student3-integration` feature branch for this implementation until it is merged; the runbook includes the clone command.

For the resource-complete current local checkout, see [independent local validation and commands](docs/local-product-retrieval-validation.md). Inputs, photographs, model files and private Sandbox configuration now reside within this checkout; the old standalone project is not required. These excluded resources remain local and are not supplied by a Git clone.
