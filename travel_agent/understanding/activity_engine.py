"""Bounded typed traversal; draft knowledge must be explicitly enabled."""

import argparse
import copy
import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "kg_data/activity/activity_kg.json"
RELATIONS = {
    "HAS_OCCASION": ("Activity", "Occasion"),
    "HAS_FORMALITY": ("Occasion", "Formality"),
    "SUGGESTS_STYLE": ("Formality", "Style"),
    "DIRECT_STYLE": ("Activity", "Style"),
    "HAS_ATTRIBUTE": ("Style", "Attribute"),
}


def normalize(value):
    return re.sub(r"[\s_-]+", " ", value.strip().casefold())


class KnowledgeGraph:
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self._validate()
        self.nodes = {n["id"]: n for n in self.data["nodes"]}
        self.aliases = {
            normalize(alias): n["id"]
            for n in self.nodes.values()
            if n["type"] == "Activity"
            for alias in [n["name"], *n["aliases"]]
        }
        self.sources = {s["id"]: s for s in self.data["sources"]}
        self.adjacency = {}
        for edge in self.data["edges"]:
            self.adjacency.setdefault(edge["source"], []).append(edge)

    @classmethod
    def load(cls, path=DATA_PATH):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _validate(self):
        def require(condition, message):
            if not condition:
                raise ValueError(message)

        require(bool(self.data.get("version")), "Missing graph version")
        for group in ("nodes", "edges", "sources"):
            rows = self.data[group]
            ids = [r["id"] for r in rows]
            require(len(ids) == len(set(ids)), f"Duplicate {group} ID")
        nodes = {n["id"]: n for n in self.data["nodes"]}
        sources = {s["id"] for s in self.data["sources"]}
        vocabulary = {}
        for node in nodes.values():
            require(
                node["type"] in {t for pair in RELATIONS.values() for t in pair},
                "Invalid node type",
            )
            require(bool(node.get("definition")), "Missing definition")
            for alias in [node["name"], *node["aliases"]]:
                key = (node["type"], normalize(alias))
                require(key not in vocabulary or vocabulary[key] == node["id"], "Ambiguous alias")
                vocabulary[key] = node["id"]
        for edge in self.data["edges"]:
            require(edge["source"] in nodes and edge["target"] in nodes, "Dangling endpoint")
            pair = (nodes[edge["source"]]["type"], nodes[edge["target"]]["type"])
            require(RELATIONS.get(edge["relation"]) == pair, "Invalid typed relationship")
            require(edge["source_id"] in sources, "Missing evidence source")
            require(
                edge["review_status"] in {"pending", "approved", "rejected"}, "Invalid review state"
            )
            require(
                edge["review_status"] != "approved" or bool(edge.get("reviewed_by")),
                "Approved assertion needs reviewer",
            )
            require(
                edge["basis"] in {"source_supported", "project_heuristic"}, "Invalid evidence basis"
            )
            require(
                bool(edge["supporting_section"]) and bool(edge["rationale"]),
                "Missing assertion evidence",
            )
            require(isinstance(edge["conditions"], dict), "Invalid conditions")
            require(set(edge["conditions"]) <= {"dress_code"}, "Unsupported condition")
            require(
                all(isinstance(v, str) and bool(v) for v in edge["conditions"].values()),
                "Invalid condition value",
            )

    def query(self, activity, context=None, *, allow_draft=False, max_depth=4):
        if type(max_depth) is not int or not 1 <= max_depth <= 4:
            raise ValueError("max_depth must be an integer between 1 and 4")
        if not isinstance(activity, str) or not activity.strip():
            raise ValueError("activity must be a non-empty string")
        if type(allow_draft) is not bool:
            raise ValueError("allow_draft must be boolean")
        if context is not None and not isinstance(context, dict):
            raise ValueError("context must be a mapping")
        context = dict(context or {})
        if "dress_code" in context:
            if not isinstance(context["dress_code"], str):
                raise ValueError("dress_code must be a string")
            context["dress_code"] = normalize(context["dress_code"])
        result = {
            "schema_version": "1.0",
            "graph_version": self.data["version"],
            "input_activity": activity,
            "activity_id": self.aliases.get(normalize(activity)),
            "status": "unknown_activity",
            "knowledge_status": "none",
            "styles": [],
            "attributes": [],
            "conclusions": [],
            "clarification": None,
            "expanded_activity_context": "",
            "constraint_level": "recommendation",
        }
        start = result["activity_id"]
        if start is None:
            return result
        node = self.nodes[start]
        allowed = node["supported_dress_codes"]
        dress_code = context.get("dress_code")
        if node["requires_dress_code"] and not dress_code:
            result.update(
                status="needs_clarification",
                clarification="Please provide the venue or invitation dress code.",
            )
            return result
        if dress_code and dress_code not in allowed:
            result.update(
                status="unsupported_context",
                clarification=(
                    "当前图谱不支持该活动与 dress code 组合；保留原要求，需人工确认或补充知识。"
                ),
            )
            return result

        conclusions = {}

        def walk(current, path_nodes, path_edges):
            if len(path_edges) >= max_depth:
                return
            for edge in self.adjacency.get(current, []):
                status = edge["review_status"]
                if status == "rejected" or (status != "approved" and not allow_draft):
                    continue
                if any(context.get(k) != normalize(v) for k, v in edge["conditions"].items()):
                    continue
                target = edge["target"]
                if target in path_nodes:
                    continue
                edges, nodes = path_edges + [edge], path_nodes + [target]
                target_node = self.nodes[target]
                if target_node["type"] in {"Style", "Attribute"}:
                    conclusion = conclusions.setdefault(
                        target,
                        {
                            "node_id": target,
                            "type": target_node["type"],
                            "value": target_node["name"],
                            "paths": [],
                        },
                    )
                    conclusion["paths"].append(
                        {
                            "nodes": nodes,
                            "edge_ids": [e["id"] for e in edges],
                            "evidence": [
                                {
                                    "edge_id": e["id"],
                                    "source_id": e["source_id"],
                                    "source_url": self.sources[e["source_id"]].get("url"),
                                    "basis": e["basis"],
                                    "review_status": e["review_status"],
                                    "supporting_section": e["supporting_section"],
                                    "rationale": e["rationale"],
                                    "conditions": e["conditions"],
                                }
                                for e in edges
                            ],
                        }
                    )
                walk(target, nodes, edges)

        walk(start, [start], [])
        result["conclusions"] = [conclusions[k] for k in sorted(conclusions)]
        for key, kind in (("styles", "Style"), ("attributes", "Attribute")):
            result[key] = sorted(c["value"] for c in conclusions.values() if c["type"] == kind)
        result["status"] = "ok" if conclusions else "no_reviewed_knowledge"
        if conclusions:
            pending = any(
                e["review_status"] != "approved"
                for c in conclusions.values()
                for p in c["paths"]
                for e in p["evidence"]
            )
            result["knowledge_status"] = "draft" if pending else "approved"
            result["expanded_activity_context"] = " ".join(
                [node["name"], *result["styles"], *result["attributes"]]
            ).replace("_", " ")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("activity")
    parser.add_argument("--dress-code")
    parser.add_argument(
        "--allow-draft", action="store_true", help="Preview assertions pending human review"
    )
    args = parser.parse_args()
    context = {"dress_code": args.dress_code} if args.dress_code else {}
    print(
        json.dumps(
            KnowledgeGraph.load().query(args.activity, context, allow_draft=args.allow_draft),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
