"""Text normalisation shared by the rule engine, the ML model and the user-correction memory."""
from __future__ import annotations

import re

_SEP = re.compile(r"[/\\\-_@.:*,()|#+=&']+")


def normalize(text: str) -> str:
    """'UPI/DR/4123/SWIGGY/YESB/swiggy@ybl' -> 'UPI DR 4123 SWIGGY YESB SWIGGY YBL'"""
    t = _SEP.sub(" ", (text or "").upper())
    return re.sub(r"\s+", " ", t).strip()


def ml_text(text: str, direction: str) -> str:
    """Input for the ML model: separators -> spaces, digits removed (reference numbers carry
    no category signal), plus a direction token so the model knows debit vs credit."""
    t = normalize(text)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"\b[A-Z]\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return f"{t} DIR_{direction.upper()}"


def memory_key(text: str, direction: str) -> str:
    """Key for remembering user corrections: digits stripped so the same merchant / person with
    different reference numbers maps to the same key."""
    t = re.sub(r"\d+", "", normalize(text))
    t = re.sub(r"\s+", " ", t).strip()
    return f"{direction}|{t}"
