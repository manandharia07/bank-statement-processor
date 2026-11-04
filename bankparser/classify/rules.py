"""Heuristic rule engine driven by config/categories.yaml (keywords + regex, first match wins)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .text import normalize

DEFAULT_RULES = Path(__file__).resolve().parents[2] / "config" / "categories.yaml"


@dataclass
class Rule:
    category: str
    direction: str | None
    keywords: list            # normalised keywords
    patterns: list            # compiled regex


class RuleEngine:
    def __init__(self, path: Path | str = DEFAULT_RULES):
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.categories: dict[str, str] = cfg["categories"]
        self.rules: list[Rule] = []
        for r in cfg["rules"]:
            if r["category"] not in self.categories:
                raise ValueError(f"Rule category '{r['category']}' is not declared under `categories`.")
            self.rules.append(Rule(
                category=r["category"],
                direction=r.get("direction"),
                keywords=[normalize(k) for k in r.get("keywords", [])],
                patterns=[re.compile(p) for p in r.get("patterns", [])],
            ))

    def match(self, description: str, direction: str):
        """Returns (category, reason) or (None, None)."""
        norm = normalize(description)
        padded = f" {norm} "
        compact = norm.replace(" ", "")
        tokens = norm.split()
        for rule in self.rules:
            if rule.direction and rule.direction != direction:
                continue
            for kw in rule.keywords:
                if f" {kw} " in padded:
                    return rule.category, f"keyword '{kw}'"
                # looser matching for joined text, e.g. 'zomato12@ybl' or 'INDIANOIL':
                #   single word -> a token STARTING with it (so 'CHARGE' never matches 'RECHARGE')
                #   multi word  -> the words glued together appear anywhere
                if " " in kw:
                    if len(kw) >= 8 and kw.replace(" ", "") in compact:
                        return rule.category, f"keyword '{kw}'"
                elif len(kw) >= 5 and any(t.startswith(kw) and t != kw for t in tokens):
                    return rule.category, f"keyword '{kw}'"
            for p in rule.patterns:
                if p.search(norm):
                    return rule.category, f"pattern /{p.pattern}/"
        return None, None
