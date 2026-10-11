# Travel Style Agent · Group 17

A shared LangChain agent project for the Intelligent Travel Style & Shopping Assistant. The repository currently provides an installable, testable development scaffold. Live weather, knowledge graph reasoning, social and product retrieval, FashionCLIP, MLP scoring, and image generation are not yet implemented.

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

The demo uses deterministic fixtures and requires no API keys. It does not perform autonomous agent decisions or demonstrate model quality. The algorithm tools currently return `not_implemented` explicitly to establish the integration boundaries.

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
