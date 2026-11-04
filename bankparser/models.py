"""Core data structures shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Word:
    """One word on a page with its position in PDF points (1/72 inch), origin top-left.
    Text PDFs and OCR'd images are both converted into this shape, so ONE parser handles both."""
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    page: int
    conf: float = 100.0           # OCR confidence (100 for native text)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass
class Line:
    words: list
    page: int

    @property
    def top(self) -> float:
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> float:
        return max(w.bottom for w in self.words)

    @property
    def x0(self) -> float:
        return min(w.x0 for w in self.words)

    @property
    def x1(self) -> float:
        return max(w.x1 for w in self.words)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class PageInfo:
    number: int                   # 1-based
    kind: str                     # "text" | "ocr"
    reason: str = ""              # why it was classified that way
    words: int = 0
    ocr_confidence: float | None = None
    rotation: int = 0
    skew: float = 0.0
    transactions: int = 0
    header_found: bool = False


@dataclass
class Transaction:
    date: date | None
    description: str
    debit: float = 0.0
    credit: float = 0.0
    balance: float | None = None
    value_date: date | None = None
    ref: str = ""
    page: int = 0
    flags: list = field(default_factory=list)      # validation notes, e.g. "balance_mismatch"
    category: str = ""
    category_source: str = ""                      # "rule" | "ml" | "user" | "none"
    confidence: float = 0.0
    category_reason: str = ""                      # human-readable explanation
    direction_unknown: bool = False                # amount found but Dr/Cr unknown yet

    @property
    def amount(self) -> float:
        return self.credit - self.debit


@dataclass
class StatementResult:
    filename: str
    account: dict
    transactions: list
    pages: list
    warnings: list = field(default_factory=list)
    reconciliation: dict = field(default_factory=dict)
    parser: str = ""              # "header-mapped" | "fallback"
    order: str = ""               # "chronological" | "reverse-chronological (re-ordered)"
