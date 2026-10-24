# Student 5 outfit preview runbook (Stage 14)

Stage 14 shows the outfit the user confirmed. The main agent reaches it through the LangChain tool `validate_and_recommend` with `action: "preview"`:

```
validate_and_recommend.invoke({"payload": {"action": "preview", ...}})
  -> travel_agent.recommendation.service.run -> _preview
  -> preview_provider.server_preview_provider()      reads PREVIEW_PROVIDER at call time
  -> Student5Orchestrator.generate_preview
  -> build_preview_request (portrait consent check)
  -> generate_preview_with_fallback
  -> preview_provider.create_live_provider            only for a generative mode
  -> OpenAIImagePreviewProvider.from_env -> OpenAI Images API
```

The provider names, `preview_source` values and `PREVIEW_*` variables are PROVISIONAL; the team has not yet agreed them as a shared interface.

## Three kinds of preview

| Kind | How it is selected | Network / cost | `preview_source` |
|---|---|---|---|
| Product cards | `preview_mode: "PRODUCT_CARDS"`, or no provider configured, or a known generation failure | None | `product_card_fallback` |
| Mock | `preview_provider: "mock"` in the payload (offline tests and demos only) | None. Returns `mock://preview/<outfit_id>/<mode>` | `mock` |
| Real OpenAI image | Server environment `PREVIEW_PROVIDER=openai` with a key and model | One paid image request per preview | `openai` |

- **Real image modes:** `FLAT_LAY`, `MANNEQUIN` and `PORTRAIT`.
- **Product cards:** never reach a provider, even when the real provider is configured.
- **Disclaimer:** every result carries a disclaimer. A generated image is a visual reference, not a virtual try-on.

## Enabling the real provider

1. Install the optional extra. The base install and default CI do not include it.

   ```bash
   uv sync --locked --extra preview
   ```

   This installs `openai`, Pillow and `requests`.

2. Set the server-side environment variables in the deployment environment, not in a tool payload, the repository or a chat:

   | Variable | Required | Meaning |
   |---|---|---|
   | `PREVIEW_PROVIDER` | yes | Exactly `openai`. This is the only switch. Unset or empty: product cards. Any other value: product cards with a configuration warning (the value is never echoed or logged). |
   | `PREVIEW_API_KEY` | yes | API key for the image service. |
   | `PREVIEW_MODEL` | yes | Image model name. |
   | `PREVIEW_ALLOWED_IMAGE_HOSTS` | no | Comma-separated exact hosts from which HTTPS product images may be downloaded. When empty, remote product images are not downloaded; only local `image_path` files are sent. |

   The service does not read a `.env` file for these values. The variables are read only when a `preview` action runs, never at import.

3. Restart the agent process.

If `PREVIEW_PROVIDER=openai` but the key, model, extra or host list is missing or invalid, the preview falls back to product cards. The same happens for the known failures that `OpenAIImagePreviewProvider` reports as `PreviewProviderError`: unusable input images, network errors, SDK API errors and invalid output.

- **Warnings:** the `warnings` field says why the preview fell back. It never contains a key, model output, base64 data, URL, query string or path.
- **Server log:** one warning line on the `travel_agent.recommendation.preview_provider` logger, for example `Live preview unavailable; using product cards: category=input type=PreviewInputError`.
  - It records only a fixed category (`configuration`, `input`, `output` or `provider`) and the exception type name.
  - It never records the exception message, so no URL, path, environment value, portrait detail, model output or API response reaches the log, whatever a provider puts in its messages.
- **Programming errors** such as `AttributeError`, `TypeError` or `AssertionError` are not turned into a fallback; they propagate so they are noticed and fixed.

## What the payload may and may not contain

- **Allowed preview fields:**
  - `confirmed_outfit` (from `confirm`)
  - `preview_mode`
  - optional `preview_provider: "mock"`
  - optional `user_portrait` with `portrait_consent`
- **Never accepted:**
  - `preview_provider: "openai"`
  - an API key, model name, allowed hosts, output directory, `.env` path, or any `PREVIEW_*` key

  Any of these returns `needs_input`, and no provider is created.

## Agent behaviour

The tool docstring and the agent system prompt tell the agent to:

- Call `preview` only after the user confirmed an outfit and explicitly asked for a preview image.
- Generate only for that one confirmed outfit, never for every Top-3 outfit.
- Tell the user that generation may incur API cost.
- Never ask for, accept or pass an API key or provider setting.

These are instructions to the LLM agent (prompt and tool contract), not code-level authorisation. The service cannot verify that the user really asked for an image, or how many times the agent calls it. They are therefore not a hard safety guarantee.

The hard, code-level checks are:

- `portrait_consent=true` is required for any portrait (enforced by the `PreviewRequest` model);
- a valid `confirmed_outfit` is required;
- the payload cannot select or configure the real provider.

## Portrait consent

`portrait_consent` is the hard check at the model level. The agent decides when to set it, but no portrait passes without it.

- **PORTRAIT mode:** needs both `user_portrait` (a local file path) and `portrait_consent: true`. The agent sets consent only after the user explicitly agrees that their portrait is sent to the image service.
- **Without consent:** a portrait without `portrait_consent: true` returns `needs_input` in any mode. This check runs before any provider is created, so the portrait is never sent.
- **Portrait handling:** portraits are only read from local files, never downloaded. They are re-encoded to PNG with EXIF removed in a temporary folder that is always deleted, and are only sent in PORTRAIT mode.
- **MANNEQUIN and FLAT_LAY:** need no portrait or portrait consent. The agent is still instructed to call them only on the user's explicit request (a prompt rule, see above).

## Cost

Each `FLAT_LAY`, `MANNEQUIN` or `PORTRAIT` preview with the real provider sends one image request (`n=1`) to the key's account. Product cards and mock previews cost nothing. The team still has to agree whose key is used and who pays.

## Output and privacy

- **Where images are saved:** generated images go to `artifacts/previews/<random-uuid>.png`, relative to the process working directory. `artifacts/` is git-ignored. The result holds only this relative path, never image bytes or base64.
- **Never commit:** generated images, portraits, `.env` files or keys.
- **Cleanup:** delete `artifacts/previews/` when the images are no longer needed. The service does not clean it up automatically.

## Tests

Default tests are offline and never call the paid API:

```bash
uv run pytest tests/recommendation
```

- **Integration tests:** `tests/recommendation/test_recommendation_preview_integration.py` covers the full path from `validate_and_recommend.invoke` to the provider. It uses a fake provider, or the real `OpenAIImagePreviewProvider` built by the default factory around a fake `openai` module. The tests that need Pillow are skipped without the `preview` extra.
- **CI safety:** `tests/recommendation/conftest.py` removes the four variables this module reads (`PREVIEW_PROVIDER`, `PREVIEW_API_KEY`, `PREVIEW_MODEL`, `PREVIEW_ALLOWED_IMAGE_HOSTS`) for every test except the live test. A runner that happens to have a key set therefore still makes no API call. Integration tests that need a configuration set it explicitly.

The live test sends one real, paid FLAT_LAY request and uses no portrait.

- **Output location:** it switches the working directory to pytest's `tmp_path` for that one test (`monkeypatch.chdir`, undone afterwards). The provider's relative default output folder `artifacts/previews/` therefore resolves inside `tmp_path`, and no global default is changed.
- **Checks:** it asserts that no new file appears in the repository's `artifacts/previews/`, and that the result is a real generated image (`preview_source` `openai`, status `SUCCESS`, a readable image file). A product-card fallback fails the test.
- **When it runs:** only when all of these hold: `RUN_LIVE_PREVIEW_TESTS=1`, `PREVIEW_PROVIDER=openai`, `PREVIEW_API_KEY` and `PREVIEW_MODEL` set, and the `preview` extra installed.

```bash
uv sync --locked --extra preview
```

```bash
RUN_LIVE_PREVIEW_TESTS=1 uv run pytest -m live_preview tests/recommendation
```

`RUN_LIVE_PREVIEW_TESTS` only controls this test. Production uses `PREVIEW_PROVIDER` alone.

### What has been verified against the real API

The automated live pytest covers only the text-input FLAT_LAY path:

- the path from the LangChain tool to the real OpenAI provider;
- `images.generate` with no reference images;
- one FLAT_LAY output;
- no portrait and no remote product images.

Two further paths were verified manually with `scripts/run_live_preview_check.py` (see [Manual acceptance check](#manual-acceptance-check)) on 2026-10-16, at commit `3cc4c9e` with OpenAI SDK 3.28:

- **FLAT_LAY:** three local garment images, sent through `images.edit` with several reference images.
- **PORTRAIT:** a local portrait, given with explicit consent, plus three local garment images.

Both runs reached the real OpenAI image API (`preview_source` `openai`, status `SUCCESS`) and produced an image that opens correctly. These are one-off manual checks, not automated regression tests.

Still **not** verified against the real API (covered only by offline tests with fake clients):

- remote product image URLs downloaded through `PREVIEW_ALLOWED_IMAGE_HOSTS`;
- a real MANNEQUIN call;
- tool selection by a real chat model (see the main-agent integration test below).

### Main-agent integration test

`tests/test_agent_preview_integration.py` starts from the shared agent, `travel_agent.agent.build_agent(model).invoke(...)`, with a scripted fake chat model and a fake preview provider. It is fully offline. It shows that the agent's technical call chain works:

agent → `validate_and_recommend` → service → server-configured provider → tool result → final reply.

It cannot show that a real LLM always judges the user's intent correctly. "Generate only when the user explicitly asks" remains a prompt and tool-contract rule, not a code-level guarantee.

## Manual acceptance check

`scripts/run_live_preview_check.py` is a manual acceptance and demo tool. It is not part of the automated regression suite: pytest runs offline by default and checks behaviour on every change, while this script is run by hand to see a real preview made from your own images and to keep a record of the result.

- **Flow:** it runs recommend → confirm → preview through `validate_and_recommend`, the same tool the shared agent calls.
- **Modes:** `--mock` (offline mock provider, no network, no cost), `--mode FLAT_LAY` and `--mode PORTRAIT` (needs `--portrait` and `--consent`). Each non-mock run sends one paid image request.
- **Checks:** `preview_source` is `openai`, status is `SUCCESS`, the output image opens, and the API key is not in the tool result. In mock mode the checks that need a real image show SKIP. Any FAIL gives a non-zero exit code.
- **Configuration:** read only from the `PREVIEW_*` environment variables of the current process. The script reads no `.env` file and reports only the names of missing variables, never their values.
- **Output:** generated images go to `artifacts/previews/` and check records to `artifacts/preview_checks/`, both git-ignored. A record holds garment categories and file names, never the API key, environment values, absolute paths or portrait details; if it would, no record is written.

```bash
uv run --locked --extra preview python -m scripts.run_live_preview_check --mode FLAT_LAY --mock --top <top image> --top-title "<title>" --bottom <bottom image> --bottom-title "<title>" --shoes <shoes image> --shoes-title "<title>"
```

Do not commit the API key, input images, portraits, generated images or check records. The script's module docstring explains how to set the variables without putting the key in the shell history.
