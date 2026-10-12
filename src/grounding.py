"""Grounding filter: drop clock times the parser wrote that the user never said.

The v3 parser sometimes adds an absolute clock reading that is not in the request
(e.g. "at 16:00" for "start the dishwasher now"). Z3 then finds a genuine clash
with the other reading and calls a POSSIBLE request impossible. A clock time is
something the user states explicitly, so it can be checked against the sentence
deterministically: if no time in the text matches it, the reading is removed
before Z3 sees it. Relative offsets are not filtered; a stated time plus/minus an offset in the same
IR also counts as stated (see ground_ir).

Times are recognised in digit form ("12:23 PM", "1:39 pm", "12 36 PM", "7 PM") and
word form ("six fifty six PM", "twelve thirty six PM", "seven o'clock").
"""

from __future__ import annotations

import copy
import re

_UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}


def _with_meridiem(h: int, m: int, mer: str | None) -> set[tuple[int, int]]:
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return set()
    if mer:
        mer = mer.replace(".", "").lower()
        if mer == "pm" and h < 12:
            h += 12
        elif mer == "am" and h == 12:
            h = 0
        return {(h, m)}
    out = {(h, m)}
    if h < 12:
        out.add((h + 12, m))  # no AM/PM given: either reading
    return out


def _words_to_numbers(text: str) -> str:
    """'six fifty six pm' -> '6 56 pm'; 'twelve thirty six' -> '12 36'."""
    tokens = re.findall(r"[a-z]+|\d+|[^a-z\d\s]", text.lower())
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in _TENS:
            val = _TENS[t]
            if i + 1 < len(tokens) and tokens[i + 1] in _UNITS and 0 < _UNITS[tokens[i + 1]] < 10:
                val += _UNITS[tokens[i + 1]]
                i += 1
            out.append(str(val))
        elif t in _UNITS:
            val = _UNITS[t]
            if t == "oh" and i + 1 < len(tokens) and tokens[i + 1] in _UNITS:  # "ten oh three" -> "10 03"
                out.append(f"{_UNITS[tokens[i + 1]]:02d}")
                i += 2
                continue
            out.append(str(val))
        else:
            out.append(t)
        i += 1
    return " ".join(out)


def times_in_text(text: str) -> set[tuple[int, int]]:
    """Every (hour 0-23, minute) the text could be stating as a clock time."""
    found: set[tuple[int, int]] = set()
    for variant in (text.lower(), _words_to_numbers(text)):
        for h, m, mer in re.findall(r"\b(\d{1,2})\s*[:.\s]\s*(\d{2})\s*(a\.?m\.?|p\.?m\.?)?", variant):
            found |= _with_meridiem(int(h), int(m), mer or None)
        for h, mer in re.findall(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)", variant):
            found |= _with_meridiem(int(h), 0, mer)
        for h in re.findall(r"\b(\d{1,2})\s*o\s*'?\s*clock", variant):
            found |= _with_meridiem(int(h), 0, None)
    return found


def _closest_device(name: str, known: list[str], cutoff: float) -> str | None:
    """Closest real device id, preferring the same room (the id prefix before the device type)."""
    import difflib

    same_room = [k for k in known if k.split("_")[0] == name.split("_")[0]]
    for pool in (same_room, known):
        hit = difflib.get_close_matches(name, pool, n=1, cutoff=cutoff)
        if hit:
            return hit[0]
    return None


def repair_device_ids(compact: dict, known_ids: list[str], cutoff: float = 0.85) -> tuple[dict, int, int]:
    """Copy of `compact` whose device ids that do not exist in the house are replaced by the closest real
    one (e.g. the parser's 'utility_laundry_dryer_1' -> 'utility_room_laundry_dryer_1'). Uses only the
    house's device list, never the answer key. Returns (ir, n_repaired, n_unresolved)."""
    known = list(known_ids)
    known_set = set(known)
    out = copy.deepcopy(compact)
    repaired = unresolved = 0

    def fix(obj: dict) -> None:
        nonlocal repaired, unresolved
        dev = obj.get("device_id")
        if not dev or dev in known_set:
            return
        hit = _closest_device(dev, known, cutoff)
        if hit:
            obj["device_id"] = hit
            repaired += 1
        else:
            unresolved += 1

    for g in out.get("goals", []):
        for r in g.get("readings", []):
            if (r.get("anchor") or {}).get("kind") == "device_event":
                fix(r["anchor"])
        for t in g.get("targets", []):
            fix(t)
    return out, repaired, unresolved


def ground_ir(compact: dict, query: str) -> tuple[dict, int]:
    """Copy of `compact` without clock readings the query does not state. Returns (ir, n_dropped).

    A clock time counts as stated if it appears in the text, or if it is a stated time
    plus/minus an offset the same IR uses ("finishes around 4:42 PM ... start 19 minutes
    after" -> 17:01): the benchmark's own labels express such derived times this way.
    """
    stated = times_in_text(query)
    offsets = {int(round(r["offset_minutes"])) for g in compact.get("goals", []) for r in g.get("readings", [])
               if isinstance(r.get("offset_minutes"), (int, float))}
    derived = set()
    for h, m in stated:
        for o in offsets:
            for t in (h * 60 + m + o, h * 60 + m - o):
                derived.add(((t // 60) % 24, t % 60))
    stated = stated | derived
    out = copy.deepcopy(compact)
    dropped = 0
    for g in out.get("goals", []):
        kept = []
        for r in g.get("readings", []):
            at = r.get("absolute_time")
            if at:
                try:
                    h, m = (int(x) for x in at.split(":")[:2])
                except ValueError:
                    dropped += 1
                    continue
                if (h, m) not in stated:
                    dropped += 1
                    continue
            kept.append(r)
        g["readings"] = kept
    return out, dropped
