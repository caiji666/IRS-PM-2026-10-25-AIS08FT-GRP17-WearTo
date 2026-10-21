# Activity KG: live Aura validation

Validated on 2026-10-19 against the existing AuraDB Free instance `travel-outfit-agent` (`051db318`). No new instance, upgrade, password reset, data deletion or knowledge approval was performed.

## Observed results

- The existing paused instance was resumed. A live database preflight found zero nodes.
- The checked-in activity seed, version `0.2.0`, was imported in one data transaction after checking that its version was absent. The uniqueness constraint was created separately.
- The actual database contains 90 activity-KG nodes and 203 relationships. All 203 assertions remain `pending`.
- Real `Neo4jKnowledgeGraph` queries matched the local engine's complete results, including evidence paths, for hiking with minimalist preference, museum visit, and dinner with an explicit `smart casual` dress code. All retained paths passed the adapter's real Cypher verification.
- The complete understanding service returned `ok` with `kg_backend=neo4j`. Expanded context, activity requirements, weather requirements and hard constraints matched the local backend. This service check used the existing synthetic-weather example and explicitly enabled draft knowledge; `context_status` remains `partial`.
- With draft knowledge disabled, hiking correctly returns `no_reviewed_knowledge`.
- The relevant offline regression tests passed: 53 tests in `test_understanding.py` and `test_activity_style_expansion.py`.

## Local configuration and reproducibility

The project's ignored `.env` contains the Aura connection settings and `ACTIVITY_KG_VERSION=0.2.0`. Existing configured values are preserved. The file mode is 0600. No credentials are included in this document or the reports. `neo4j==6.4.1` was installed into the existing Python 3.11 environment, within the project's declared optional dependency range; dependency declarations and the lockfile were not changed.

Run the existing project CLI from the repository root (the saved example explicitly selects `kg_backend=neo4j`):

```bash
.venv/bin/python -m travel_agent.understanding.demo \
  --input artifacts/neo4j-validation/neo4j-service-input.json \
  --output artifacts/neo4j-validation/standard-cli-result.json
```

This existing CLI loads `.env` normally. It returned `ok` using the unmodified project backend and standard system resolver; no validation-helper import, socket patch or process-local DNS fallback was used. The service still requires an explicit `kg_backend=neo4j` payload; the offline default was not changed. Local reports are in `artifacts/neo4j-validation/standard-connection-report.json` and `standard-cli-result.json`.

The earlier `verify_activity_aura.py` helper is retained locally as a historical import/validation aid. Its DNS fallback is not needed by the normal project startup and was not added to application code.

## DNS investigation and final state

The first validation encountered failed system resolution immediately after resuming the paused Aura instance. Public DNS independently returned the same valid address, and the initial check used a process-local fallback while retaining TLS verification. The precise cause of the initial resolver discrepancy was not proven.

On the follow-up check, ordinary `socket.getaddrinfo` resolved the Aura hostname successfully. macOS showed DHCP-provided DNS servers, `192.168.1.1` and `192.168.68.1`; `networksetup -getdnsservers Wi-Fi` confirmed no manually configured DNS servers. No DNS-server setting, hosts-file entry, proxy, browser security setting, application resolver or certificate-validation option was changed. There is therefore no configuration change to roll back.

The unmodified `Neo4jKnowledgeGraph.from_environment()` connected to the existing 90-node/203-relationship graph. Hiking returned `ok` with `path_query_verified=true`; both the direct understanding-service call and the existing project CLI returned `ok`. Connections retain the original `neo4j+s` URI and certificate verification.

Refreshing Aura's previously failed Query page restored its native connection, confirmed by the UI's `Connected ... via neo4j+s://` status. No custom DNS fallback, IP address or HTTPS fallback is used for this connection.

## Native Aura Query verification

The original Chrome Aura tab is left on the native Query interface:

- Project: `63efc291-c7dd-40e8-8b6d-93159118673d`.
- Existing instance: `travel-outfit-agent` / `051db318`.
- A real count query returned **90 nodes and 203 relationships**.
- A real bounded hiking query returned **15 nodes, 17 relationships and 17 path records** in Graph view.
- The hiking query was saved in Aura under **Hiking · Activity KG 0.2.0**.
- Native screenshots are saved locally as `artifacts/neo4j-validation/aura-native-counts.jpg` and `aura-native-hiking.jpg`.

The earlier local snapshot viewer remains an optional export, not the native verification result. Read-only queries for other cases are in [neo4j-activity-demo.cypher](neo4j-activity-demo.cypher).

The runtime continues to use Python for conditional reasoning and Neo4j for storage and retained-path verification. The service example still uses synthetic weather and explicitly enabled draft knowledge, so `context_status=partial` is expected. No knowledge was approved; place KG runtime data was not imported or switched to Neo4j. The observed recovery does not guarantee future DNS or free-instance uptime.
