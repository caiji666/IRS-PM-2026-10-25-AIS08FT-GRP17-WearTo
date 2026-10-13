"""Optional Neo4j storage and bounded path traversal; never import/approve data."""

import json
import os

from travel_agent.understanding.activity_engine import RELATIONS, KnowledgeGraph


class Neo4jKnowledgeGraph(KnowledgeGraph):
    def __init__(self, data, uri, user, password, database):
        super().__init__(data)
        self.connection = (uri, user, password, database)

    @classmethod
    def from_environment(cls):
        from neo4j import GraphDatabase

        uri = os.environ.get("NEO4J_URI")
        password = os.environ.get("NEO4J_PASSWORD")
        if not uri or not password:
            raise ValueError("Set NEO4J_URI and NEO4J_PASSWORD locally for the Neo4j backend")
        user = os.environ.get("NEO4J_USER", "neo4j")
        database = os.environ.get("NEO4J_DATABASE", "neo4j")
        version = os.environ.get("ACTIVITY_KG_VERSION", KnowledgeGraph.load().data["version"])
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=database, default_access_mode="READ") as session:
                nodes = session.run(
                    "MATCH (n:Student1KG) WHERE n.graph_version=$version "
                    "RETURN properties(n) AS props, labels(n) AS labels",
                    version=version,
                ).data()
                edges = session.run(
                    "MATCH (a:Student1KG)-[r]->(b:Student1KG) "
                    "WHERE a.graph_version=$version AND b.graph_version=$version "
                    "AND r.graph_version=$version RETURN a.id AS source, b.id AS target, "
                    "type(r) AS relation, properties(r) AS props",
                    version=version,
                ).data()
        if not nodes:
            raise ValueError("The requested graph version is not imported into Neo4j")
        node_types = {t for pair in RELATIONS.values() for t in pair}
        parsed_nodes, parsed_edges, sources = [], [], {}
        for row in nodes:
            props = dict(row["props"])
            props.pop("uid", None)
            props.pop("graph_version", None)
            types = set(row["labels"]) & node_types
            if len(types) != 1:
                raise ValueError("Neo4j node has an invalid domain type")
            props["type"] = types.pop()
            parsed_nodes.append(props)
        for row in edges:
            props = dict(row["props"])
            props.pop("graph_version", None)
            props["conditions"] = json.loads(props.pop("conditions_json"))
            source = json.loads(props.pop("source_record_json"))
            sources[source["id"]] = source
            props.update({key: row[key] for key in ("source", "target", "relation")})
            parsed_edges.append(props)
        data = {
            "version": version,
            "nodes": parsed_nodes,
            "edges": parsed_edges,
            "sources": list(sources.values()),
        }
        return cls(data, uri, user, password, database)

    def query(self, activity, context=None, *, allow_draft=False, max_depth=4, preferred_styles=()):
        from neo4j import GraphDatabase

        result = super().query(
            activity,
            context,
            allow_draft=allow_draft,
            max_depth=max_depth,
            preferred_styles=preferred_styles,
        )
        result["backend"] = "neo4j"
        if result["status"] != "ok":
            return result
        uri, user, password, database = self.connection
        # Typed traversal and dress-code conditions are validated by the shared
        # engine; Neo4j confirms each retained path actually exists in this DB.
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=database, default_access_mode="READ") as session:
                rows = session.run(
                    "MATCH path=(a:Student1KG {uid:$uid})-[*1..4]->(b:Student1KG) "
                    "WHERE length(path)<=$depth AND all(r IN relationships(path) WHERE "
                    "r.graph_version=$version AND type(r) IN $relations AND "
                    "(r.review_status='approved' OR ($draft AND r.review_status='pending'))) "
                    "RETURN [r IN relationships(path) | r.id] AS edge_ids LIMIT 2000",
                    uid=f"student1:{self.data['version']}:{result['activity_id']}",
                    version=self.data["version"],
                    depth=max_depth,
                    relations=list(RELATIONS),
                    draft=allow_draft,
                ).data()
        actual = {tuple(row["edge_ids"]) for row in rows}
        for conclusion in result["conclusions"]:
            for path in conclusion["paths"]:
                if tuple(path["edge_ids"]) not in actual:
                    raise ValueError("Neo4j graph changed or traversal is incomplete; reload graph")
        result["path_query_verified"] = True
        return result
