"""
Parsing of dates and money amounts as they appear on Indian bank statements,
including common OCR mistakes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTHS["sept"] = 9

_NUM_DATE = re.compile(r"^(\d{1,2})[/\-.\s](\d{1,2})[/\-.\s](\d{2,4})$")
_ISO_DATE = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$")
_TXT_DATE = re.compile(r"^(\d{1,2})[\s\-/.,]*([A-Za-z]{3,9})[\s\-/.,']*(\d{2,4})$")
_OCR_DIGITS = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                             "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"})


def _year(y: str) -> int:
    v = int(y)
    if len(y) == 2:
        v += 2000 if v < 70 else 1900
    return v


def _safe(y: int, m: int, d: int) -> date | None:
    if not (1990 <= y <= 2100):
        return None
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(text: str) -> date | None:
    """Parse a day-first date (Indian convention). Returns None when the text is not a date."""
    if not text:
        return None
    s = text.strip().strip(",;:.'`|_~-\"")       # OCR specks around the date, e.g. '11/04/24.'
    if len(s) < 6 or len(s) > 20:
        return None
    for candidate in (s, _ocr_fix_date(s)):
        if candidate is None:
            continue
        m = _NUM_DATE.match(candidate)
        if m:
            return _safe(_year(m.group(3)), int(m.group(2)), int(m.group(1)))
        m = _ISO_DATE.match(candidate)
        if m:
            return _safe(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = _TXT_DATE.match(candidate)
        if m:
            mon = MONTHS.get(m.group(2).lower()[:4]) or MONTHS.get(m.group(2).lower()[:3])
            if mon:
                return _safe(_year(m.group(3)), mon, int(m.group(1)))
    return None


def _ocr_fix_date(s: str) -> str | None:
    """'O1/O4/2O24' -> '01/04/2024'. Only applied to purely numeric-looking dates."""
    if re.fullmatch(r"[\dOoDQIl|iSsBZz/\-.\s]{6,12}", s) and re.search(r"\d", s):
        return s.translate(_OCR_DIGITS)
    return None


def find_date_prefix(text: str):
    """Find a date at the start of a line (used by the header-less fallback parser).
    Returns (date, length_of_match) or (None, 0)."""
    tokens = text.split()
    for n in (3, 2, 1):                      # '01 Apr 2024' | '01-Apr 2024' | '01/04/2024'
        if len(tokens) >= n:
            cand = " ".join(tokens[:n])
            d = parse_date(cand)
            if d:
                return d, n
    return None, 0


# ------------------------------------------------------------------ amounts
@dataclass
class Amount:
    value: float          # always >= 0 unless an explicit minus/brackets was present
    hint: str | None      # 'dr' / 'cr' when a suffix/prefix said so


_AMOUNT_TOKEN = re.compile(
    r"^\(?[-+]?(?:rs\.?|inr|₹)?\s?[\dOoIlS][\dOoIlS,. ]*[.,]\s?\d\d\s?(?:cr|dr|c|d)?\.?\)?-?$", re.I)


def looks_like_amount(text: str) -> bool:
    """True for '1,23,456.78', '450.00 Cr', '(12.50)', '-99.00' ... (requires 2 decimals)."""
    t = text.strip()
    return bool(t) and bool(_AMOUNT_TOKEN.match(t)) and sum(ch.isdigit() for ch in t) >= 3


def parse_amount(text: str) -> Amount | None:
    if not text:
        return None
    s = text.strip()
    if not s or s in {"-", "--", "—", "0", "nil", "NIL"}:
        return None
    hint = None
    low = s.lower().replace("₹", "").replace("inr", "").replace("rs.", "").replace("rs", "").strip()
    m = re.search(r"\(?\s*(cr|dr|c|d)\.?\s*\)?$", low)
    if m and re.search(r"\d", low[: m.start()]):
        hint = {"c": "cr", "d": "dr"}.get(m.group(1), m.group(1))
        low = low[: m.start()].strip()
    m = re.match(r"^(cr|dr)\.?\s*", low)
    if m:
        hint = m.group(1)
        low = low[m.end():]
    neg = False
    if low.startswith("(") and low.endswith(")"):
        neg, low = True, low[1:-1]
    if low.startswith("-") or low.endswith("-"):
        neg, low = True, low.strip("-")
    low = low.strip("+ ").replace(" ", "")
    low = low.translate(str.maketrans({"o": "0", "i": "1", "l": "1", "s": "5", "|": "1"}))
    if not re.fullmatch(r"[\d,.]+", low) or not re.search(r"\d", low):
        return None
    # decide which separator is the decimal point
    if "." in low:
        head, _, tail = low.rpartition(".")
        if len(tail) != 2 and "," in tail:   # weird -> give up
            return None
        low = head.replace(".", "").replace(",", "") + "." + tail
    elif re.search(r",\d\d$", low) and not re.search(r",\d{3}$", low):
        head, _, tail = low.rpartition(",")   # OCR read '.' as ','
        low = head.replace(",", "") + "." + tail
    else:
        low = low.replace(",", "")
    try:
        v = float(low)
    except ValueError:
        return None
    return Amount(-v if neg else v, hint)


def fmt_inr(v: float | None) -> str:
    """12345678.9 -> '1,23,45,678.90' (Indian grouping) for display."""
    if v is None:
        return ""
    neg = v < 0
    whole, frac = f"{abs(v):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + "." + frac
