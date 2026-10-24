"""Owner approval must be traceable and must not change underlying knowledge."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from travel_agent.understanding.activity_engine import KnowledgeGraph

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "travel_agent/understanding/kg_data"
AUDIT = ROOT / "docs/kg-review-2026-10-22"
REVIEW_FIELDS = {"review_status", "reviewed_by", "reviewed_at"}


def test_activity_review_ledger_preserves_facts_and_matches_current_snapshot():
    ledger = json.loads((AUDIT / "activity-review-ledger.json").read_text())
    data = KnowledgeGraph.load().data
    for event in ledger["events"]:
        before = {k: v for k, v in event["before"].items() if k not in REVIEW_FIELDS}
        after = {k: v for k, v in event["after"].items() if k not in REVIEW_FIELDS}
        assert before == after
        current = next(r for r in data[event["collection"]] if r["id"] == event["id"])
        assert current == event["after"]
        assert current["review_status"] == "approved"
        assert current["reviewed_by"] == ledger["reviewer"]
    assert len(ledger["events"]) == sum(len(data[k]) for k in ("nodes", "edges", "sources"))
    assert KnowledgeGraph.load().query("hiking")["knowledge_status"] == "approved"


def test_place_review_events_cover_current_facts_without_changing_source_verification():
    ledger = json.loads((AUDIT / "place-review-events.json").read_text())
    expected = Counter()
    actual = Counter()
    for event in ledger["events"]:
        assert event["decision"] == "approve" and event["reviewer"] == "project_owner"
        assert event["note"] and event["evidence_ref"] and event["spec_hash"]
        assert json.loads(event["patch"]) == {}
        expected[event["table_name"], event["content_hash"]] += 1
    for path in (SNAPSHOT / "place").glob("*.csv"):
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            assert row["review_status"] == "approved" and row["reviewed_by"] == "project_owner"
            facts = {k: v for k, v in row.items() if k not in REVIEW_FIELDS | {"updated_at"}}
            encoded = json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            actual[path.stem, hashlib.sha256(encoded.encode()).hexdigest()] += 1
    assert actual == expected


def test_runtime_manifest_fingerprints_match_all_delivered_files():
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == expected, name
