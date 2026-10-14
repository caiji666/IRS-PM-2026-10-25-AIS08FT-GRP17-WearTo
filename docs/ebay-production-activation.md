# eBay Production activation and notification lifecycle

This branch adds a local-tested notification receiver and an authenticated pull worker. It does **not** activate an eBay keyset, deploy a public endpoint, guarantee Browse Production access or certify compliance. Sandbox retrieval remains usable without lifecycle configuration. Existing shared contracts and other modules are unchanged.

## Architecture

```text
eBay GET challenge -> HTTPS callback -> SHA-256 response
eBay signed POST -> ECDSA/SHA1 validation -> encrypted durable SQLite events -> 204
Mac worker -> authenticated HTTPS claim -> registered output/cache cleanup -> ACK
```

The receiver validates the raw payload, or compact JSON compatible with the official Node SDK, against the exact ECDSA/SHA1 notification protocol. It fetches public keys from the selected eBay Notification API using an application token and caches them for one hour. This SHA1 use is required by the notification protocol, not the challenge (which uses SHA256). A public official SDK test vector is included with its Apache license attribution.

- Callback GET computes `SHA256(challenge_code + verification_token + exact_endpoint_url)`.
- Callback POST accepts only the account-deletion topic and account identifiers after signature validation. Body limit is 256 KB. Invalid signatures return 412. Dependency/persistence failures return 503 so eBay retries. It acknowledges only after successful durable persistence.
- The queue encrypts identity payloads with a separate Fernet key. Pending events survive restart. Notification IDs deduplicate retries; conflicting duplicate identities are rejected.
- Each configured consumer has its own bearer token and five-minute claim lease. A notification stays pending until **every configured data holder** acknowledges cleanup. Failure uses bounded exponential backoff; expired claims can be retried. Identity payloads are erased from the active queue after all acknowledgments, leaving event IDs, fingerprints and delivery status.
- `/healthz` checks database availability without exposing identities. Authenticated `/internal/status` reports queue counts and oldest pending age. Logs omit bodies, secrets, usernames and challenge values. Configure external alerts for failed health checks, cleanup failures and growing queue age before going live.

## Local preparation

From the repository root, with Python 3.11 and uv:

```bash
uv sync --locked --extra products --extra ebay-notifications
uv run python -m travel_agent.products.notifications init-secrets
```

This creates `.secrets/ebay-notifications.env` with file mode 0600 and fresh verification token, queue encryption key and Mac worker token. It never prints secret values or overwrites an existing file. View this file privately when configuring deployment; do not paste secrets into chat or GitHub. Existing eBay App ID and Cert ID are separate from these generated secrets.

For a local-only listener, set the variables in `examples/notifications/service.env.example` using real local secrets. The configured endpoint must be the intended public HTTPS URL (the challenge hash uses it), even though the test listener binds only to loopback. Use a private `.env` file with mode 0600:

```bash
set -a
source .secrets/ebay-notifications.env
source .secrets/ebay-service.env
set +a
uv run python -m travel_agent.products.notifications serve-local --port 8765
```

In another terminal:

```bash
curl 'http://127.0.0.1:8765/healthz'
curl 'http://127.0.0.1:8765/ebay/account-deletion?challenge_code=local-test'
```

GET verification needs no network access. A signed POST needs valid eBay signature and public-key retrieval; unsigned arbitrary JSON is rejected. Test signatures use synthetic keys or the official public vector; do not install a signature bypass in the deployed service.

## Deployment preparation (not deployed yet)

A Dockerfile and Gunicorn configuration are in `deploy/ebay-notifications/`. The Docker build context excludes secrets, data, model weights and caches; it installs only the notification extra, not FashionCLIP/Torch.

```bash
docker build -f deploy/ebay-notifications/Dockerfile -t ebay-notifications .
```

Use a stable HTTPS origin and TLS termination with a valid certificate. Mount a persistent disk at `/var/lib/ebay`, writable by UID 10001. Run one service replica for this SQLite design; use a managed database if horizontal scaling is required. Ephemeral storage, temporary tunnels, or an intermittently sleeping instance are not an operational notification setup. Configure request rate limits and monitoring at the proxy/platform. Do not enable access logs that expose queries or authorization headers.

Set the environment variables from `examples/notifications/service.env.example`, the generated queue key and worker token map, and your Production App ID/Cert ID. Keep the queue encryption key stable across restarts; changing it without a migration makes pending events unreadable. Do not remove consumers with pending deliveries. Keep registered data holders' workers active, including when the Mac is offline: events accumulate and require cleanup when it reconnects.

After deployment:

1. Verify HTTPS `/healthz` and the GET challenge response.
2. Configure the exact endpoint and matching verification token in eBay Marketplace Account Deletion settings; leave exemption off.
3. Save and send a test notification. Verify signature validation and persisted delivery, not just HTTP reachability.
4. Run the Mac worker, check completion and queue status, then confirm the keyset no longer says disabled.
5. Separately test Production OAuth, keyword search and image search permissions. Key activation does not guarantee Buy API approval.

## Mac data lifecycle

Copy `examples/notifications/worker.example.json` to `.secrets/ebay-worker.json`, replace its placeholders, and restrict permissions to 0600. The token must match the service's `mac` token. Worker URLs require HTTPS and never follow redirects. `--allow-local` permits only loopback HTTP for local transport testing.

```bash
uv run python -m travel_agent.products.notifications worker \
  --config .secrets/ebay-worker.json
```

Without `--apply`, this reports status only and does not claim or delete events. To operate the lifecycle worker:

```bash
uv run python -m travel_agent.products.notifications worker \
  --config .secrets/ebay-worker.json --apply --watch --interval 30
```

`--apply` removes registered generated files/directories, not original wardrobe photos or model weights. Use a persistent supervised process and inspect errors; command failures keep events queued. Acknowledgment failures are safe to retry because the local completion journal deduplicates cleanup.

Before a Production retrieval run:

```bash
export EBAY_MANAGED_ROOT="$PWD"
uv run python -m travel_agent.products.retrieval.ebay \
  --credentials .secrets/ebay-production.json run \
  --pt data/team_input/phase_2_output.pt \
  --text data/team_input/expanded_fashion_context.txt \
  --context data/team_input/context.json \
  --search-mode hybrid --output-dir runs/ebay_production
```

Production persistence now requires an explicit managed root. The run directory and encoder cache are registered before acquisition. Seller usernames/user IDs/EIAS identifiers available in summaries/details are linked using local keyed hashes. Previously deleted identities are blocked. Generation and cleanup use an OS lock to avoid races, and old results cannot be saved after deletion invalidates them. Registration DB and journal live in `.cache/ebay-lifecycle/`.

The initial implementation conservatively invalidates **all registered eBay run outputs, registered aggregate exports and registered embedding cache directories** on each verified deletion, rather than trusting incomplete API seller IDs. This includes raw responses, local product photos, vectors, indexes and copies of ranking/wardrobe bundles in those reports. Other sellers' generated output may also be invalidated and must be regenerated. Original wardrobe stores/photos, model weights, credentials and repository code are protected. It never processes filesystem paths supplied in a remote notification.

Only registered artifacts are managed. Production shared-tool output can also exist in memory or be copied by downstream modules: release references when notified and copy reports with the lock-protected `export-report` command below, or use `save_report` for the pipeline report. Do not distribute untracked files to teammates; each holder needs a configured consumer/cleanup integration. This branch does not retroactively discover arbitrary historical copies, remote exports, backups, clipboard contents or downstream databases. Inventory any such copies before Production use.

Logical file deletion and SQLite secure-delete cannot promise forensic erasure on SSDs, snapshots or backup systems. Establish encrypted storage, retention/backup deletion, downstream propagation and deployment monitoring before claiming operational compliance. Pending identity payloads and encryption keys must not be placed in uncontrolled backups. Never mark a failed cleanup complete.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
ITSA_REAL_MODEL_TEST=1 uv run pytest -q
```

Notification tests require the optional notification extra and skip cleanly in the default environment. Tests cover official signature interoperability, tamper rejection, challenge output, durable deduplication, all-consumer completion, encryption, retries, safe paths, invalidated reports, HTTP auth and end-to-end synthetic cleanup. Deployment verification is recorded below; automated tests remain separate from live API and deployment checks.

References: [eBay account deletion guide](https://developer.ebay.com/develop/guides/sell/marketplace-user-account-deletion), [official notification SDK](https://github.com/eBay/event-notification-nodejs-sdk), [Buy API access requirements](https://developer.ebay.com/api-docs/buy/static/buy-requirements.html).


Create a managed local downstream export (copies the full report, with handoff under `bundle`):

```bash
uv run python -m travel_agent.products.notifications export-report \
  --root "$PWD" --source runs/ebay_production/result.json \
  --target runs/handoff/production-result.json
```

This registers the exported JSON for cleanup under the same lock as deletion. Its image paths still refer to the managed run; copying it to another computer requires registering that holder separately.

## Local validation record (2026-10-14)

- Base main: `1928e25`; feature branch: `feat/ebay-production-activation`.
- Full environment with real FashionCLIP enabled: 1374 passed, 17 skipped.
- Default locked CI environment: 1039 passed, 182 skipped (optional model/service tests excluded).
- Notification-only environment: all 29 notification tests passed; Torch and Pillow absent.
- Actual loopback WSGI HTTP smoke: GET challenge 200; unsigned POST 412; official SDK signed fixture POST 204; authenticated Mac pull, synthetic artifact cleanup and ACK completed.
- Ruff lint/format, dependency lock sync and shared offline demo passed.
- At the initial local-validation stage, Docker was unavailable. The subsequent Render deployment built and ran the image successfully; see the live deployment record below.

Authenticated status includes failed deliveries and expired leases. Status-only worker calls return exit 1 and emit an error when cleanup failures exist or queue age exceeds `--max-pending-age` (default 3600 seconds); monitoring can alert on this exit code. Continuous workers report errors and retain failed deliveries for retry. Configure the deployment platform's actual alert destination separately.


## Live deployment and Production verification (2026-10-14)

- The notification service branch was pushed independently of main and deployed on Render Starter with a 1 GB persistent disk mounted at `/var/lib/ebay`.
- HTTPS health and GET challenge verification passed. Production notification settings were saved successfully in eBay, and the disabled keyset notice disappeared.
- A real eBay test notification passed signature verification and entered the encrypted queue. After a Render service restart, the pending event remained available. The authenticated Mac worker completed it and acknowledged it; subsequent status reported one done event, zero failed deliveries and zero expired leases. No generated artifacts existed when this test cleanup ran.
- Local Production OAuth authentication succeeded without displaying or persisting the access token.
- Live hybrid retrieval using the previously supplied team inputs completed 13 search requests: 99 image-search candidates and 140 keyword-search candidates, 239 unique candidates, and 40 encoded products. All four target categories returned five products. One keyword request encountered a network error; the overall run still completed.
- The result contained one saved owned item. Social references were re-encoded locally (`reference_mode: recompute`); this run did not establish compatibility of the old anonymous upstream vectors. Scoring used equal context/social weights and maximum similarity across social references.
- Runtime results are local at `runs/ebay_production/result.json`; the downstream handoff is its `bundle` field. Three generated artifact locations were registered for lifecycle cleanup. Runtime results, credentials, user inputs, downloaded images and model weights are excluded from Git.
- The worker was verified in single-pass mode. Continuous operation still requires the documented watch command or a supervised process on each registered data holder. This is functional verification, not a retrieval-quality evaluation or a claim of complete operational compliance.

The latest fetched Student 2 code is already present in this branch. Git does not contain newly generated `.pt` outputs; a new handoff must be supplied privately or generated with the upstream API credentials before testing those exact updated artifacts.
