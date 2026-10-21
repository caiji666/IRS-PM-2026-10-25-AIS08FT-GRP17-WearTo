// Activity KG 0.2.0 only. Read-only demonstration queries.
// Pending assertions are shown for demonstration, not approved knowledge.

// 1. Whole graph (90 nodes, 203 relationships when the snapshot is intact).
MATCH p=(a:Student1KG)-[r]->(b:Student1KG)
WHERE a.graph_version='0.2.0' AND b.graph_version='0.2.0'
  AND r.graph_version='0.2.0'
RETURN p;

// 2. Hiking: alternative aesthetics and independent functional requirements.
MATCH p=(a:Student1KG:Activity {id:'activity:hiking', graph_version:'0.2.0'})-[*1..2]->(b:Student1KG)
WHERE all(r IN relationships(p) WHERE r.graph_version='0.2.0'
  AND r.review_status IN ['pending','approved'] AND r.conditions_json='{}')
RETURN p;

// 3. Museum visit: a bounded activity subgraph; no invented venue dress code.
MATCH p=(a:Student1KG:Activity {id:'activity:museum_visit', graph_version:'0.2.0'})-[*1..2]->(b:Student1KG)
WHERE all(r IN relationships(p) WHERE r.graph_version='0.2.0'
  AND r.review_status IN ['pending','approved'] AND r.conditions_json='{}')
RETURN p;

// 4. Dinner with an explicitly supplied smart casual dress code.
MATCH p=(a:Student1KG:Activity {id:'activity:dinner', graph_version:'0.2.0'})-[*1..4]->(b:Student1KG)
WHERE all(r IN relationships(p) WHERE r.graph_version='0.2.0'
  AND r.review_status IN ['pending','approved']
  AND (r.conditions_json='{}' OR r.conditions_json CONTAINS '"dress_code": "smart casual"'))
RETURN p;

// 5. Inspect source evidence and review state rather than approving assertions.
MATCH (a:Student1KG)-[r]->(b:Student1KG)
WHERE r.graph_version='0.2.0'
RETURN a.name AS source, type(r) AS relation, b.name AS target,
       r.id AS assertion_id, r.review_status AS review_status,
       r.basis AS basis, r.conditions_json AS conditions,
       r.source_record_json AS source_evidence
ORDER BY assertion_id;
