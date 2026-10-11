# Contributing

Use English for documentation, code comments, examples, issues, commit messages, and pull requests.

1. Accept the repository invitation, clone the repository, and run `uv sync --locked`.
2. Before starting a task, run `git switch main` and `git pull --ff-only`.
3. Create a short-lived branch, such as `git switch -c feat/product-retrieval`.
4. Keep business logic separate from LangChain tool wrappers. Use the shared input and output schemas.
5. Provide fixed input/output examples and meaningful tests for your module. Do not download models or call external APIs at module import time.
6. Before committing, run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest`.
7. Push your branch and open a pull request. Describe behavior changes, validation results, data dependencies, and interface impacts.
8. Squash merge after a peer review. Changes to shared schemas require input from affected module owners.

Keep `main` runnable. Default CI must not require API keys, GPUs, or live external services. Run offline training separately and record the code commit, dataset splits, configuration, random seeds, and model version.

Enforcement of branch protection depends on the repository's GitHub plan. Follow the pull request workflow even when protection is not enabled.
