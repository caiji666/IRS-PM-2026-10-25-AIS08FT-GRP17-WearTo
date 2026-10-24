"""Read CSVs and evaluate scoped requirements. Python standard library only."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def split(value):
    return [s for s in (value or "").split(";") if s]


def truth(value):
    return str(value).lower() == "true"


class KG:
    def __init__(self, data_dir=None):
        self.root = Path(data_dir or ROOT / "kg_data/place")
        self.tables = {}
        for p in sorted(self.root.glob("*.csv")):
            with p.open(encoding="utf-8-sig", newline="") as f:
                self.tables[p.stem] = list(csv.DictReader(f))
        self.vocab = json.loads((self.root / "vocabulary.json").read_text())
        self.places = {p["place_id"]: p for p in self.tables["places"]}
        self.sources = {s["source_id"]: s for s in self.tables["sources"]}
        self.requirements = {r["req_id"]: r for r in self.tables["requirements"]}
        self.activity_defs = {r["activity_id"]: r for r in self.tables["activities"]}
        self.items = {r["item_id"]: r for r in self.tables["items"]}

    def usable(self, row, draft):
        return row.get("review_status") == "approved" or (
            draft and row.get("review_status") == "pending"
        )

    def resolve(self, value):
        if value in self.places:
            return value
        if "local:" + value in self.places:
            return "local:" + value
        found = {
            a["place_id"]
            for a in self.tables["aliases"]
            if a["alias"].casefold() == value.casefold()
        }
        if len(found) != 1:
            raise ValueError(f"Place is unknown or its alias is ambiguous: {value}")
        return found.pop()

    def ancestors(self, pid):
        result = []
        while pid:
            if pid in result:
                raise ValueError("part_of cycle")
            result.append(pid)
            pid = self.places[pid]["part_of"]
        return result

    def proof(self, table, r):
        return dict(table=table, record=r, source_url=self.sources[r["source"]]["url"])

    def query(
        self,
        places,
        month=None,
        activities=(),
        enter_interior=False,
        gender="unknown",
        cold_sensitive=False,
        allow_draft=False,
    ):
        if month is not None and (
            isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12
        ):
            raise ValueError("month must be an integer from 1 to 12 or null")
        if gender not in ("unknown", "male", "female"):
            raise ValueError("gender must be unknown, male or female")
        if not places:
            raise ValueError("At least one explicit place is required")
        if set(activities) - set(self.vocab["activity"]):
            raise ValueError("An activity is outside the supported place vocabulary")
        pids = list(dict.fromkeys(self.resolve(p) for p in places))
        contexts = {"enter_interior": enter_interior, "cold_sensitive": cold_sensitive}
        result = {
            "schema_version": "0.1.1",
            "knowledge_status": "draft" if allow_draft else "reviewed_only",
            "month": month,
            "places": [],
            "requirements": [],
            "conditional_requirements": [],
            "missing_information": [],
            "search_queries": [],
            "limitations": [
                "Climate normals are not trip-date weather forecasts.",
                "Item categories are candidates; product and outfit compliance remains unverified.",
                "Venue rules cover adults; student/special-event conditions need review.",
            ],
        }
        acc = {}
        conditional = []

        def add(req, strength, pid, proofs):
            if not self.usable(self.requirements[req], allow_draft):
                return
            if req not in acc:
                acc[req] = {
                    "req_id": req,
                    "name": self.requirements[req]["name"],
                    "strength": strength,
                    "evidence_paths": [],
                    "candidate_items": [],
                }
            rank = {"nice": 0, "should": 1, "must": 2}
            if rank[strength] > rank[acc[req]["strength"]]:
                acc[req]["strength"] = strength
            acc[req]["evidence_paths"].append({"place_id": pid, "steps": proofs})

        def condition(r, pid, base, table):
            req_id = r.get("req_id", r.get("then_req"))
            if not self.usable(self.requirements[req_id], allow_draft):
                return False
            missing = [c for c in split(r.get("condition")) if not contexts.get(c, False)]
            scope = r.get("gender_scope", "all")
            if scope != "all":
                if gender == "unknown":
                    missing.append("gender_scope=" + scope)
                elif scope != gender:
                    return False
            if missing:
                conditional.append(
                    {
                        "place_id": pid,
                        "req_id": r.get("req_id", r.get("then_req")),
                        "strength": r["strength"],
                        "conditions_needed": missing,
                        "evidence_path": base + [self.proof(table, r)],
                    }
                )
                return False
            return True

        selected_any = set()
        for pid in pids:
            p = self.places[pid]
            if not self.usable(p, allow_draft):
                result["missing_information"].append(
                    {"place_id": pid, "reason": "no_reviewed_place"}
                )
                continue
            chain = self.ancestors(pid)
            # Do not traverse a rejected/unreviewed hierarchy link under the chosen review policy.
            scoped = []
            for ancestor in chain:
                if not self.usable(self.places[ancestor], allow_draft):
                    break
                scoped.append(ancestor)
            available = [
                a
                for a in self.tables["place_activity"]
                if a["place_id"] == pid
                and self.usable(a, allow_draft)
                and self.usable(self.activity_defs[a["activity_id"]], allow_draft)
            ]
            selected = [a for a in available if a["activity_id"] in activities]
            selected_any.update(a["activity_id"] for a in selected)
            pr = {
                "place_id": pid,
                "name": p["name_zh"],
                "available_activities": [a["activity_id"] for a in available],
                "selected_activities": [a["activity_id"] for a in selected],
                "climate_profiles": [],
                "season_events": [],
                "optional_styles": [],
            }
            result["places"].append(pr)
            base = [self.proof("places", p)]
            # Select the nearest outdoor profile set; prefer the requested month to annual defaults.
            # Indoor venue profiles are separate; exclude outdoor climate for indoor-only visits.
            profiles = []
            for c in self.tables["climate"]:
                if (
                    c["place_id"] == pid
                    and c["profile_kind"] == "indoor_venue"
                    and self.usable(c, allow_draft)
                ):
                    pr["climate_profiles"].append(c)
            if p["indoor_outdoor"] != "indoor":
                for ancestor in scoped:
                    candidates = [
                        c
                        for c in self.tables["climate"]
                        if c["place_id"] == ancestor
                        and c["profile_kind"] == "outdoor_climatology"
                        and self.usable(c, allow_draft)
                        and (ancestor == pid or truth(c["inherit_to_descendants"]))
                    ]
                    exact = [
                        c for c in candidates if month is not None and c["month"] == str(month)
                    ]
                    annual = [c for c in candidates if not c["month"]]
                    profiles = exact or annual
                    if profiles:
                        break
                pr["climate_profiles"].extend(profiles)
                if not profiles:
                    result["missing_information"].append(
                        {
                            "place_id": pid,
                            "reason": "month_or_climate_missing",
                            "detail": (
                                "Provide month/local weather; "
                                "do not transfer climate across unrelated locations."
                            ),
                        }
                    )
            if available and not selected:
                result["missing_information"].append(
                    {"place_id": pid, "reason": "select_actual_activities"}
                )
            triggers = []
            for c in profiles:
                path = [
                    self.proof("places", self.places[a])
                    for a in scoped[: scoped.index(c["place_id"]) + 1]
                ] + [self.proof("climate", c)]
                triggers.extend(("climate_tag", tag, path) for tag in split(c["climate_tags"]))
            for e in self.tables["season_events"]:
                if (
                    not self.usable(e, allow_draft)
                    or e["place_id"] not in scoped
                    or (e["place_id"] != pid and not truth(e["inherit_to_descendants"]))
                ):
                    continue
                if p["indoor_outdoor"] == "indoor" and e["place_id"] != pid:
                    continue
                if month is None or not e["start_month"] or not e["end_month"]:
                    continue
                start, end = int(e["start_month"]), int(e["end_month"])
                active = (
                    (start <= month <= end) if start <= end else (month >= start or month <= end)
                )
                if active:
                    pr["season_events"].append(e)
                    path = [
                        self.proof("places", self.places[a])
                        for a in scoped[: scoped.index(e["place_id"]) + 1]
                    ] + [self.proof("season_events", e)]
                    triggers.extend(("climate_tag", tag, path) for tag in split(e["climate_tags"]))
            triggers.extend(
                ("activity", a["activity_id"], base + [self.proof("place_activity", a)])
                for a in selected
            )
            triggers.extend(("environment_tag", tag, base) for tag in split(p["environment_tags"]))
            triggers += [("place_id", pid, base), ("place_type", p["place_type"], base)]
            for r in self.tables["rules"]:
                if not self.usable(r, allow_draft):
                    continue
                for itype, value, path in triggers:
                    if (
                        r["if_type"] == itype
                        and r["if_value"] == value
                        and condition(r, pid, path, "rules")
                    ):
                        add(r["then_req"], r["strength"], pid, path + [self.proof("rules", r)])
            for d in self.tables["place_dresscode"]:
                if (
                    d["place_id"] == pid
                    and self.usable(d, allow_draft)
                    and condition(d, pid, base, "place_dresscode")
                ):
                    add(d["req_id"], d["strength"], pid, base + [self.proof("place_dresscode", d)])
            for s in self.tables["place_style"]:
                if s["place_id"] == pid and self.usable(s, allow_draft):
                    pr["optional_styles"].append(s)
        unmatched = set(activities) - selected_any
        if unmatched:
            result["missing_information"].append(
                {
                    "reason": "activity_not_available_at_selected_places",
                    "activities": sorted(unmatched),
                }
            )
        # One explicit second stage, rather than unbounded arbitrary graph reachability.
        first_pass = list(acc.values())
        for r in self.tables["rules"]:
            if r["if_type"] != "requirement" or not self.usable(r, allow_draft):
                continue
            for req in first_pass:
                if req["req_id"] == r["if_value"]:
                    for path in list(req["evidence_paths"]):
                        if condition(r, path["place_id"], path["steps"], "rules"):
                            add(
                                r["then_req"],
                                r["strength"],
                                path["place_id"],
                                path["steps"] + [self.proof("rules", r)],
                            )
        for req in acc.values():
            for m in self.tables["requirement_item"]:
                if (
                    m["req_id"] == req["req_id"]
                    and self.usable(m, allow_draft)
                    and self.usable(self.items[m["item_category"]], allow_draft)
                ):
                    req["candidate_items"].append(
                        {
                            "item_id": m["item_category"],
                            "mapping": m,
                            "item": self.items[m["item_category"]],
                        }
                    )
            if not req["candidate_items"]:
                req["action_or_constraint_only"] = True
        result["requirements"] = sorted(acc.values(), key=lambda r: r["req_id"])
        result["conditional_requirements"] = conditional
        for p in result["places"]:
            related = [
                r
                for r in result["requirements"]
                if any(x["place_id"] == p["place_id"] for x in r["evidence_paths"])
            ]
            place = self.places[p["place_id"]]
            for r in related:
                result["search_queries"].append(
                    {
                        "zh": f"{place['name_zh']} {r['name']} 旅行穿搭",
                        "en": f"{place['name_en']} {r['req_id'].replace('_', ' ')} travel outfit",
                        "req_id": r["req_id"],
                        "place_id": p["place_id"],
                    }
                )
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("places", nargs="+")
    ap.add_argument("--month", type=int)
    ap.add_argument("--activity", action="append", default=[])
    ap.add_argument("--enter-interior", action="store_true")
    ap.add_argument("--cold-sensitive", action="store_true")
    ap.add_argument("--gender", choices=["unknown", "male", "female"], default="unknown")
    ap.add_argument("--allow-draft", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    try:
        result = KG().query(
            a.places,
            a.month,
            a.activity,
            a.enter_interior,
            a.gender,
            a.cold_sensitive,
            a.allow_draft,
        )
    except ValueError as e:
        ap.error(str(e))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if a.output:
        a.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
