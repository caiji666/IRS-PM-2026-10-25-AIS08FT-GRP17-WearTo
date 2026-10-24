"""Apply the owner's recorded review metadata to the existing Activity KG and compare backends."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from travel_agent.understanding.activity_engine import KnowledgeGraph
from travel_agent.understanding.neo4j_backend import Neo4jKnowledgeGraph


def main():
    load_dotenv()
    local = KnowledgeGraph.load()
    version = local.data["version"]
    out = Path("evaluation/results/2026-10-22")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "graph_version": version,
        "operation": "review metadata only; no creation/deletion",
        "cases": [],
    }
    try:
        with GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
            connection_timeout=15,
        ) as driver:
            with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:

                def update(tx):
                    nodes = tx.run(
                        "MATCH (n:Student1KG) WHERE n.graph_version=$v "
                        "RETURN n.id AS id, properties(n) AS properties",
                        v=version,
                    ).data()
                    edges = tx.run(
                        "MATCH (a:Student1KG)-[r]->(b:Student1KG) "
                        "WHERE a.graph_version=$v AND b.graph_version=$v AND r.graph_version=$v "
                        "RETURN r.id AS id, properties(r) AS properties",
                        v=version,
                    ).data()
                    if {n["id"] for n in nodes} != {n["id"] for n in local.data["nodes"]} or {
                        r["id"] for r in edges
                    } != {r["id"] for r in local.data["edges"]}:
                        raise ValueError("Graph identity mismatch; review update aborted")
                    # Persist old properties before any database write; never log credentials.
                    (out / "neo4j-before-review.json").write_text(
                        json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2)
                        + "\n"
                    )
                    node_rows = [
                        {
                            "id": n["id"],
                            **{k: n[k] for k in ("review_status", "reviewed_by", "reviewed_at")},
                        }
                        for n in local.data["nodes"]
                    ]
                    edge_rows = [
                        {
                            "id": e["id"],
                            **{k: e[k] for k in ("review_status", "reviewed_by", "reviewed_at")},
                            "source_record_json": json.dumps(
                                local.sources[e["source_id"]], ensure_ascii=False
                            ),
                        }
                        for e in local.data["edges"]
                    ]
                    tx.run(
                        "UNWIND $rows AS row MATCH (n:Student1KG {id:row.id,graph_version:$v}) "
                        "SET n.review_status=row.review_status,n.reviewed_by=row.reviewed_by,"
                        "n.reviewed_at=row.reviewed_at",
                        rows=node_rows,
                        v=version,
                    ).consume()
                    tx.run(
                        "UNWIND $rows AS row MATCH (:Student1KG)-[r]->(:Student1KG) "
                        "WHERE r.graph_version=$v AND r.id=row.id "
                        "SET r.review_status=row.review_status,r.reviewed_by=row.reviewed_by,"
                        "r.reviewed_at=row.reviewed_at,r.source_record_json=row.source_record_json",
                        rows=edge_rows,
                        v=version,
                    ).consume()
                    return {"nodes": len(nodes), "relationships": len(edges)}

                report["updated"] = session.execute_write(update)
        remote = Neo4jKnowledgeGraph.from_environment()
        for activity, context, pref, depth in [
            ("hiking", {}, ["minimalist"], 4),
            ("museum_visit", {}, [], 4),
            ("dinner", {"dress_code": "smart casual"}, [], 4),
            ("dinner", {"dress_code": "casual"}, [], 1),
        ]:
            expected = local.query(activity, context, preferred_styles=pref, max_depth=depth)
            actual = remote.query(activity, context, preferred_styles=pref, max_depth=depth)
            comparable = {
                k: v for k, v in actual.items() if k not in ("backend", "path_query_verified")
            }
            report["cases"].append(
                {
                    "activity": activity,
                    "depth": depth,
                    "local_equal": comparable == expected,
                    "path_query_verified": actual.get("path_query_verified"),
                    "status": actual["status"],
                    "knowledge_status": actual["knowledge_status"],
                }
            )
        report["status"] = (
            "passed"
            if all(c["local_equal"] and c["path_query_verified"] for c in report["cases"])
            else "failed"
        )
    except Exception as exc:
        report.update(status="unavailable_or_failed", error_type=type(exc).__name__)
    (out / "neo4j-reviewed-validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
