"""Source-grounded English facts/edits supplement the small Qwen extractor.

These are constrained lexical rules, not a replacement semantic model. Unsupported
or ambiguous edits are left for clarification and human review. No test IDs or
expected answers are used by the production parser.
"""

import re
from datetime import datetime

ACTIVITIES = {
    "sightseeing": ("sightseeing", "city walk", "city walking", "exploring the city"),
    "museum visit": ("museum visit", "museum visits", "visiting museums", "museum", "museums"),
    "hiking": ("hiking", "hike"),
    "dinner": ("dinner", "dining"),
    "business meeting": ("business meeting", "business meetings"),
    "wedding guest": ("wedding guest", "wedding"),
}
STYLES = {
    "minimalist": ("minimalist", "minimalism"),
    "casual": ("casual",),
    "sporty": ("sporty",),
    "bohemian": ("bohemian", "boho"),
    "classic": ("classic",),
    "streetwear": ("streetwear",),
    "maximalist": ("maximalist", "maximalism"),
    "preppy": ("preppy", "ivy style"),
    "artistic": ("artistic", "artsy"),
    "utility": ("utility", "utilitarian"),
    "romantic": ("romantic",),
    "retro": ("retro", "vintage inspired"),
}
ITEMS = {
    "boots": ("boots",),
    "skirt": ("skirt", "skirts"),
    "long coat": ("long coat", "long coats"),
    "high heels": ("high heels",),
    "raincoat": ("raincoat", "raincoats"),
    "dress": ("dress", "dresses"),
    "shorts": ("shorts",),
    "sandals": ("sandals",),
    "sneakers": ("sneakers",),
}
CURRENCY = {
    "SGD": ("SGD", "Singapore dollars"),
    "USD": ("USD", "US dollars"),
    "GBP": ("GBP", "British pounds", "pounds"),
    "EUR": ("EUR", "euros"),
    "JPY": ("JPY", "yen"),
    "KRW": ("KRW", "won"),
    "CNY": ("CNY", "RMB"),
    "MYR": ("MYR", "ringgit"),
    "THB": ("THB", "baht"),
    "VND": ("VND", "dong"),
    "IDR": ("IDR", "rupiah"),
}
CODES = (
    "smart casual",
    "business casual",
    "business formal",
    "semi formal",
    "semi-formal",
    "black tie",
    "black-tie",
    "white tie",
    "cocktail",
    "casual",
    "formal",
)
NEGATIVE = re.compile(r"\b(?:not|no|without|cancel|exclude|avoid)\b|don't|do not", re.I)


def sanitize(message):
    warnings = []

    def replace(match):
        fragment = match.group(1)
        if re.search(
            r"\b(?:ignore|override|schema|invent|output budget|return budget)\b", fragment, re.I
        ):
            warnings.append("Ignored instructions inside quoted material")
            return " "
        return fragment

    text = re.sub(r'["“]([^"”]+)["”]', replace, message)
    return text, warnings


def clauses(text):
    return re.split(r"[;!?]|(?<!\d)\.(?!\d)|\bbut\b", text, flags=re.I)


def occurrences(text, terms):
    pattern = r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b"
    return list(re.finditer(pattern, text, re.I))


def positive(clause, match):
    prefix = clause[: match.start()]
    # A comma resets a negative clause unless it continues an exclusion list.
    return not NEGATIVE.search(prefix.rsplit(",", 1)[-1])


def mentions(text, vocabulary, positive_only=False):
    found = []
    for canonical, terms in vocabulary.items():
        if any(
            not positive_only or positive(clause, hit)
            for clause in clauses(text)
            for hit in occurrences(clause, terms)
        ):
            found.append(canonical)
    return found


def canonical_activity(value):
    return next(
        (key for key, terms in ACTIVITIES.items() if value.strip().casefold() in terms),
        value.strip().casefold(),
    )


def date_facts(text):
    found = [(m.start(), m.group()) for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text)]
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    for match in re.finditer(
        rf"\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b", text, re.I
    ):
        try:
            value = datetime.strptime(" ".join(match.groups()), "%B %d %Y").date().isoformat()
        except ValueError:
            raise ValueError("Invalid explicit English calendar date") from None
        found.append((match.start(), value))
    found.sort()
    facts, ordered = {}, []
    for position, value in found:
        if value not in ordered:
            ordered.append(value)
        prefix = text[max(0, position - 35) : position].casefold()
        if re.search(
            r"(?:return(?:ing)?|end(?: date)?|back|until|\bto)\s*(?:is|on|:)?\s*$", prefix
        ):
            facts["end_date"] = value
        elif re.search(
            r"(?:depart(?:ure|ing)?|start(?: date)?|from|leave|leaving)\s*(?:is|on|:)?\s*$", prefix
        ):
            facts["start_date"] = value
    if len(ordered) == 2:
        facts.setdefault("start_date", ordered[0])
        facts.setdefault("end_date", ordered[1])
    elif len(ordered) > 2:
        raise ValueError("Multiple date ranges require clarification")
    return facts


def currency_in(text):
    codes = [code for code, terms in CURRENCY.items() if occurrences(text, terms)]
    if len(codes) > 1:
        raise ValueError("Multiple currencies in the same budget clause require clarification")
    return codes[0] if codes else None


def money_facts(text):
    facts = {}
    for clause in clauses(text):
        if not re.search(r"\bbudget\b|\bspending limit\b", clause, re.I):
            continue
        tail = re.split(r"\bbudget\b|\bspending limit\b", clause, flags=re.I)[-1]
        number = re.search(r"(?<!\w)[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)", tail)
        if number:
            value = number.group().replace(",", "")
            facts["budget"] = float(value) if "." in value else int(value)
            code = currency_in(tail)
            if code:
                facts["currency"] = code
    if "currency" not in facts:
        for clause in clauses(text):
            if re.search(r"\bcurrency\s+(?:is|to|should be)\b", clause, re.I):
                code = currency_in(clause)
                if code:
                    facts["currency"] = code
    # A complete amount/currency reply is explicit even without the word budget.
    currency_terms = "|".join(re.escape(term) for terms in CURRENCY.values() for term in terms)
    amount = r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?"
    reply = re.fullmatch(
        rf"\s*(?:({amount})\s*({currency_terms})|({currency_terms})\s*({amount}))\s*[.!]?\s*",
        text,
        re.I,
    )
    if reply:
        value = (reply.group(1) or reply.group(4)).replace(",", "")
        facts.update(
            budget=float(value) if "." in value else int(value), currency=currency_in(text)
        )
    elif re.fullmatch(rf"\s*(?:{currency_terms})\s*[.!]?\s*", text, re.I):
        facts["currency"] = currency_in(text)
    return facts


def destination_fact(text, candidate):
    # Bound extraction to explicit place-introducing syntax, or a place heading.
    matches = re.finditer(
        r"\b(?:going to|travel(?:l)?ing to|trip (?:is|to)|destination(?:\s+is|\s+to|:))\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        text,
    )
    for match in matches:
        prefix = text[: match.start()].rsplit("but", 1)[-1].rsplit(".", 1)[-1]
        if not NEGATIVE.search(prefix):
            return match.group(1).strip()
    match = re.match(
        r"\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*[,;]|"
        r"\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+from\b",
        text,
    )
    if match:
        return (match.group(1) or match.group(2)).strip()
    value = candidate.get("destination")
    non_places = {
        term for vocab in (STYLES, ACTIVITIES, ITEMS) for terms in vocab.values() for term in terms
    }
    if (
        isinstance(value, str)
        and value.strip().casefold() == text.strip(" .!\n").casefold()
        and value.strip().casefold() not in non_places
    ):
        return value.strip()
    if (
        isinstance(value, str)
        and value
        and re.search(
            r"\b(?:in|to|visit|visiting|destination)\s+" + re.escape(value) + r"\b", text, re.I
        )
        and any(
            positive(clause, hit)
            for clause in clauses(text)
            for hit in occurrences(clause, (value,))
        )
    ):
        return value.strip()
    return None


def exclusion_items(text):
    result = []
    for clause in clauses(text):
        cue = re.search(
            r"\b(?:exclude|avoid)\b|(?:do not|don't)\s+(?:include|want)|\bno\s+", clause, re.I
        )
        if cue:
            tail = clause[cue.end() :]
            for item, terms in ITEMS.items():
                if any(
                    not (item == "dress" and re.match(r"\s+code\b", tail[hit.end() :], re.I))
                    for hit in occurrences(tail, terms)
                ):
                    result.append(item)
    return list(dict.fromkeys(result))


def style_facts(text, candidate):
    found = []
    for canonical, terms in STYLES.items():
        if text.strip(" .!\n").casefold() in terms:
            return [canonical]
    for clause in clauses(text):
        if not re.search(r"\b(?:prefer|like|want|style)\b", clause, re.I):
            continue
        if re.search(r"\bdress code\b", clause, re.I) and not re.search(
            r"\b(?:prefer|like)\b", clause, re.I
        ):
            continue
        found.extend(mentions(clause, STYLES, positive_only=True))
        for value in candidate.get("style", []):
            if (
                isinstance(value, str)
                and value
                and value.casefold() not in {"travel", "clothes", "outfits"}
                and re.search(r"\b(?:prefer|like|style)\b", clause, re.I)
                and any(positive(clause, hit) for hit in occurrences(clause, (value,)))
            ):
                found.append(value.strip().casefold())
    return list(dict.fromkeys(found))


def dress_code_facts(text, activities):
    codes = {}
    for clause in clauses(text):
        if not re.search(r"\bdress code\b|\brequires?\b", clause, re.I):
            continue
        for activity in activities:
            for term in ACTIVITIES.get(activity, (activity,)):
                # Require a local activity/code binding, not mere co-occurrence
                # in a sentence that lists several activities.
                match = re.search(
                    rf"\b{re.escape(term)}\b\s+(?:has\s+(?:a\s+)?|"
                    r"(?:dress code)(?:\s+is|:)?\s*|requires?\s+)"
                    r"([a-z][a-z -]*?)(?:\s+dress code|[,;.!]|$)",
                    clause,
                    re.I,
                )
                if match:
                    value = match.group(1).strip().replace("-", " ").casefold()
                    if value not in {"no", "none", "no dress code"}:
                        codes[activity] = value
                else:
                    match = re.search(
                        rf"dress code\s+for\s+{re.escape(term)}\s+is\s+(.+)$", clause, re.I
                    )
                    if match:
                        codes[activity] = match.group(1).strip().casefold()
    return codes


def english_facts(message, candidate):
    text, warnings = sanitize(message)
    facts = date_facts(text) | money_facts(text)
    destination = destination_fact(text, candidate)
    if destination:
        facts["destination"] = destination
    activities = mentions(text, ACTIVITIES, positive_only=True)
    for value in candidate.get("activities", []):
        if (
            isinstance(value, str)
            and value.casefold() not in {"travel", "clothes", "outfits"}
            and value not in activities
            and re.search(r"\b(?:plan|activities|activity)\b", text, re.I)
            and any(
                positive(clause, hit)
                for clause in clauses(text)
                for hit in occurrences(clause, (value,))
            )
        ):
            activities.append(canonical_activity(value))
    if activities:
        facts["activities"] = list(dict.fromkeys(activities))
    styles = style_facts(text, candidate)
    if styles:
        facts["style"] = styles
    excluded = exclusion_items(text)
    if excluded:
        facts["excluded_items"] = excluded
    codes = dress_code_facts(text, activities)
    if codes:
        facts["dress_codes"] = codes
    return facts, text, warnings


def grounded_english(candidate, message, previous=None):
    facts, text, warnings = english_facts(message, candidate.get("set", candidate))
    if previous is None:
        return facts, warnings
    # Resolve short answers only where the accumulated request makes the slot
    # unambiguous. Do not guess a date role or overwrite a known budget from a number.
    missing_dates = [
        field for field in ("start_date", "end_date") if getattr(previous, field) is None
    ]
    if len(missing_dates) == 1 and re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*[.!]?\s*", text):
        facts[missing_dates[0]] = text.strip(" .!\n")
    if previous.budget is None and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", text):
        facts["budget"] = float(text) if "." in text else int(text)
    if re.search(r"\b(?:why|explain|reason)\b", text, re.I) and re.search(
        r"\boutfit|recommend", text, re.I
    ):
        return {"intent": "REQUEST_EXPLANATION"}, warnings
    selection = re.search(r"\b(?:confirm|choose|select|take|go with)\b", text, re.I)
    if selection and NEGATIVE.search(text[: selection.start()]):
        raise ValueError("A negated outfit choice needs clarification")
    if selection:
        action = {"intent": "CONFIRM_OUTFIT"}
        outfit = re.search(r"\boutfit_[A-Za-z0-9_-]+\b", text)
        ordinal = re.search(r"\b(first|second|third|1st|2nd|3rd)\s+(?:outfit|option)\b", text, re.I)
        position = re.search(r"\b(?:outfit|option|number)\s*([123])\b", text, re.I)
        if outfit:
            action["outfit_id"] = outfit.group()
        if ordinal:
            action["display_position"] = {
                "first": 1,
                "second": 2,
                "third": 3,
                "1st": 1,
                "2nd": 2,
                "3rd": 3,
            }[ordinal.group(1).casefold()]
        elif position:
            action["display_position"] = int(position.group(1))
        return action, warnings
    patch = {"intent": "MODIFY_REQUIREMENTS", "set": {}, "add": {}, "remove": {}}
    if re.fullmatch(r"\s*(?:no (?:style )?preferences?|any style)\s*[.!]?\s*", text, re.I):
        patch["remove"]["style"] = []
    if re.fullmatch(r"\s*(?:no activities|no planned activities)\s*[.!]?\s*", text, re.I):
        patch["remove"]["activities"] = []
    skip_optional = bool(
        re.fullmatch(
            r"\s*(?:skip(?: optional (?:fields|details))?|nothing else)\s*[.!]?\s*", text, re.I
        )
    )
    for field in ("destination", "start_date", "end_date", "budget", "currency"):
        if field in facts:
            patch["set"][field] = facts[field]
    if re.search(r"\b(?:cheaper|lower price|less expensive)\b", text, re.I):
        patch["set"]["price_preference"] = "lower"
    for clause in clauses(text):
        active = mentions(clause, ACTIVITIES)
        if re.search(r"\b(?:cancel|remove|drop)\b", clause, re.I) and active:
            patch["remove"].setdefault("activities", []).extend(active)
        elif re.search(r"\badd\b", clause, re.I) and active:
            patch["add"].setdefault("activities", []).extend(active)
        elif active and re.search(r"\b(?:activities|activity|plan)\b", clause, re.I):
            patch["set"]["activities"] = mentions(clause, ACTIVITIES, positive_only=True)
        elif active and not re.search(r"\b(?:keep|unchanged|same)\b", clause, re.I):
            additions = mentions(clause, ACTIVITIES, positive_only=True)
            if additions:
                patch["add"].setdefault("activities", []).extend(additions)
        if re.search(r"\b(?:clear|reset)\b.*\b(?:style|preferences)\b", clause, re.I):
            patch["remove"]["style"] = []
        elif re.search(r"\b(?:remove|drop|cancel)\b.*\bstyle\b", clause, re.I):
            patch["remove"]["style"] = mentions(clause, STYLES)
        elif re.search(r"\b(?:prefer|like|style)\b", clause, re.I) and facts.get("style"):
            patch["set"]["style"] = facts["style"]
        if re.search(r"\b(?:clear|reset)\b.*\bexclusions?\b", clause, re.I):
            patch["remove"]["excluded_items"] = []
        elif re.search(r"\b(?:remove|drop)\b.*\bexclusions?\b", clause, re.I):
            patch["remove"]["excluded_items"] = mentions(clause, ITEMS)
    if facts.get("excluded_items"):
        patch["add"]["excluded_items"] = facts["excluded_items"]
    if facts.get("style") and "style" not in patch["remove"] and "style" not in patch["set"]:
        patch["set"]["style"] = facts["style"]
    codes = dict(previous.dress_codes)
    codes.update(facts.get("dress_codes", {}))
    for clause in clauses(text):
        if re.search(r"\b(?:remove|clear|cancel)\b.*\bdress code\b", clause, re.I):
            for activity in mentions(clause, ACTIVITIES):
                codes.pop(activity, None)
    if codes != previous.dress_codes:
        patch["set"]["dress_codes"] = codes
    for group in ("add", "remove"):
        for field, values in patch[group].items():
            patch[group][field] = list(dict.fromkeys(values))
    if not skip_optional and not any(patch[group] for group in ("set", "add", "remove")):
        raise ValueError(
            "The English follow-up does not specify a supported clear edit; clarify it"
        )
    return patch, warnings
