"""Explicit date normalization; ambiguous dates remain questions, never guesses."""

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from travel_agent.understanding.english_evidence import sanitize

MONTHS = {
    name: i
    for i, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        1,
    )
}
MONTHS.update({name[:3]: value for name, value in list(MONTHS.items())})
MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def normalize_dates(message, *, awaiting=None, current=None, today=None):
    """Return normalized source, explicit ISO slot facts, and a clarification.

    The clock is injectable and defaults to the Singapore date at message processing.
    Yearless dates and next-week expressions require confirmation, with a concrete
    proposed year/date. Durations ask for a return date rather than defining nights.
    """
    today = today or datetime.now(ZoneInfo("Asia/Singapore")).date()
    message, _ = sanitize(message)
    current = current or {}
    hits = []
    issues = []

    def add(match, year, month, day):
        try:
            value = date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            issues.append("That calendar date is invalid. What date did you mean?")
            value = None
        hits.append((match.start(), match.end(), value))

    for m in re.finditer(r"(?<!\d)(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?(?!\d)", message):
        add(m, *m.groups())
    for pattern, order in (
        (rf"\b({MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b", (2, 0, 1)),
        (rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_PATTERN})\.?[,]?\s+(\d{{4}})\b", (2, 1, 0)),
    ):
        for m in re.finditer(pattern, message, re.I):
            groups = m.groups()
            add(m, groups[order[0]], MONTHS[groups[order[1]].lower()], groups[order[2]])

    def occupied(start, end):
        return any(start < b and end > a for a, b, _ in hits)

    # Day/month/year is accepted only when its order is mathematically unambiguous.
    for m in re.finditer(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?!\d)", message):
        if occupied(m.start(), m.end()):
            continue
        a, b, y = map(int, m.groups())
        if a <= 12 and b <= 12 and a != b:
            issues.append(f"Does {m.group()} mean {y}-{a:02d}-{b:02d} or {y}-{b:02d}-{a:02d}?")
            hits.append((m.start(), m.end(), None))
        else:
            add(m, y, b if a > 12 else a, a if a > 12 else b)
    for m in re.finditer(r"\b(today|tomorrow|day after tomorrow)\b|今天|明天|后天", message, re.I):
        offset = {
            "today": 0,
            "tomorrow": 1,
            "day after tomorrow": 2,
            "今天": 0,
            "明天": 1,
            "后天": 2,
        }[m.group().lower()]
        hits.append((m.start(), m.end(), (today + timedelta(days=offset)).isoformat()))
    for m in re.finditer(
        r"\bnext\s+(" + "|".join(WEEKDAYS) + r")\b|下(?:周|星期)([一二三四五六日天])", message, re.I
    ):
        weekday = WEEKDAYS.index(m[1].lower()) if m[1] else min("一二三四五六日天".index(m[2]), 6)
        target = today + timedelta(days=7 - today.weekday() + weekday)
        issues.append(
            f"Do you mean {target.isoformat()} for “{m.group()}”? Please confirm with the date."
        )
        hits.append((m.start(), m.end(), None))
    for m in re.finditer(
        rf"\b(?:{MONTH_PATTERN})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b|\d{{1,2}}月\d{{1,2}}日?",
        message,
        re.I,
    ):
        if not occupied(m.start(), m.end()):
            issues.append(
                "Which year are these dates in? You can reply with just the year; "
                "for a cross-year trip, specify both full dates."
            )
            hits.append((m.start(), m.end(), None))
    hits.sort()
    facts = {}
    if len(hits) > 2:
        issues.append(
            "Which departure and return dates should I use? Please specify one trip date range."
        )
    else:
        for index, (start, end, value) in enumerate(hits):
            if value is None:
                continue
            prefix = message[max(0, start - 35) : start].lower()
            if re.search(
                r"(?:return(?:ing)?|end(?: date)?|back|until|\bto|返回|回程|回来|到|至)"
                r"\s*(?:is|on|:|是)?\s*$",
                prefix,
            ):
                field = "end_date"
            elif re.search(
                r"(?:depart(?:ure|ing)?|start(?: date)?|from|leave|leaving|出发|去程)"
                r"\s*(?:is|on|:|是)?\s*$",
                prefix,
            ):
                field = "start_date"
            elif len(hits) == 2:
                field = ("start_date", "end_date")[index]
            elif awaiting in ("start_date", "end_date"):
                field = awaiting
            else:
                missing = [key for key in ("start_date", "end_date") if not current.get(key)]
                field = missing[0] if len(missing) == 1 else None
            if field:
                if field in facts and facts[field] != value:
                    issues.append("Please specify one departure and one return date.")
                facts[field] = value
            else:
                issues.append(f"Is {value} your departure date or return date?")
    duration = re.search(
        r"\b(?:for|stay(?:ing)?)\s+(?:\d+|one|two|three|four|five|six|seven)"
        r"\s+(?:days?|nights?)\b|(?:玩|待|住|停留)\s*[\d一二三四五六七八九十]+\s*[天晚]",
        message,
        re.I,
    )
    if duration and not facts.get("end_date") and not current.get("end_date"):
        issues.append(
            "What is your return date? Please specify it so days and nights are not confused."
        )
    start = facts.get("start_date", current.get("start_date"))
    end = facts.get("end_date", current.get("end_date"))
    if start and end and end < start:
        facts = {}
        issues.insert(
            0, "The return date is before departure. What departure and return dates should I use?"
        )
    normalized = message
    for start, end, _ in reversed(hits):
        normalized = normalized[:start] + " " + normalized[end:]
    for field, value in facts.items():
        label = "Start date" if field == "start_date" else "End date"
        normalized += f". {label} is {value}."
    return normalized, facts, next(iter(issues), None)
