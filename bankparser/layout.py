"""
Layout-agnostic transaction table parser.

Input: Words with (x, y) positions (from the text layer OR from OCR - same format).
Output: Transactions.

How it works (no bank-specific code):
  1. Group words into visual lines by their vertical centre.
  2. Find the table HEADER line by matching column-name synonyms
     ("Narration" / "Particulars" / "Description", "Withdrawal" / "Debit" / "Dr", ...).
     Two-line headers ("Txn" over "Date") are merged.
  3. Find COLUMN BOUNDARIES from the data itself: vertical whitespace "gutters" that
     run through all transaction rows. Each gutter-separated band is mapped to the header
     cell it overlaps. This copes with left-aligned text and right-aligned numbers.
  4. Walk the lines: a line with a valid date in the Date column starts a new transaction;
     lines with only description text are continuations of a wrapped narration
     (attached to the row above or below using the biggest vertical gap);
     "Opening balance", "B/F", "C/F", totals and footers are recognised and skipped.
  5. If no header is found at all, a FALLBACK parser reads "date ... description ... amount balance"
     lines, so unknown layouts still produce output.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import numpy as np

from .models import Line, Transaction, Word
from .normalize import find_date_prefix, looks_like_amount, parse_amount, parse_date

# Order matters: the first matching rule wins ("Value Date" before "Date", "Withdrawal Amt" before "Amount").
ROLE_RULES = [
    ("drcr", re.compile(r"^(dr\s*/\s*cr|cr\s*/\s*dr|dr\s*cr|d\s*/\s*c|c\s*/\s*d|type|txn type|cr dr)$")),
    ("value_date", re.compile(r"\bvalue\b|\bval\s*dt\b")),
    ("balance", re.compile(r"\bbal(ance)?\b")),
    ("debit", re.compile(r"\b(withdrawals?|withdrawl|debits?|dr|paid out|dr amount)\b")),
    ("credit", re.compile(r"\b(deposits?|credits?|cr|paid in|receipts?)\b")),
    ("date", re.compile(r"\b(date|dt|txn dt|tran dt)\b")),
    ("description", re.compile(r"\b(narration|description|particulars|details|remarks|transaction details)\b")),
    ("ref", re.compile(r"\b(chq|cheque|ref|reference|instrument|utr)\b")),
    ("amount", re.compile(r"\b(amount|amt)\b")),
]
NUMERIC_ROLES = {"debit", "credit", "amount", "balance"}

STOP_RE = re.compile(
    r"^\s*(statement summary|account summary|summary of|page\s*\d+(\s*(of|/)\s*\d+)?\s*$|"
    r"this is a (computer|system)[\s-]*generated|\*+\s*end of|end of statement|grand total|totals?\b|"
    r"abbreviations|disclaimer|important (note|information))", re.I)
SPECIAL_RE = re.compile(
    r"(opening\s+bal(ance)?|balance\s+b\s*/?\s*f|brought\s+f(or)?w(ar)?d|\bb\s*/\s*f\b|balance\s+forward|"
    r"carried\s+f(or)?w(ar)?d|balance\s+c\s*/?\s*f|\bc\s*/\s*f\b|closing\s+bal(ance)?)", re.I)


@dataclass
class HeaderCell:
    role: str          # one of the roles above, or "ignore"
    x0: float
    x1: float
    text: str


@dataclass
class Header:
    cells: list
    top: float
    bottom: float


@dataclass
class Column:
    role: str
    left: float        # region boundaries used to assign words
    right: float


# ------------------------------------------------------------------ geometry helpers
def group_lines(words: list[Word]) -> list[Line]:
    if not words:
        return []
    heights = [w.height for w in words if w.height > 0]
    tol = (statistics.median(heights) if heights else 8.0) * 0.55
    lines, cur, cy = [], [], None
    for w in sorted(words, key=lambda w: w.cy):
        if cur and abs(w.cy - cy) <= tol:
            cur.append(w)
            cy = (cy * (len(cur) - 1) + w.cy) / len(cur)
        else:
            if cur:
                lines.append(cur)
            cur, cy = [w], w.cy
    if cur:
        lines.append(cur)
    return [Line(sorted(ws, key=lambda w: w.x0), ws[0].page) for ws in lines]


def char_width(words: list[Word]) -> float:
    vals = [(w.x1 - w.x0) / len(w.text) for w in words if len(w.text) >= 3 and w.x1 > w.x0]
    return statistics.median(vals) if vals else 4.0


def split_cells(line: Line, gap: float):
    """Split a line into cells wherever the horizontal gap between words exceeds `gap`."""
    cells, cur = [], [line.words[0]]
    for w in line.words[1:]:
        if w.x0 - cur[-1].x1 > gap:
            cells.append(cur)
            cur = [w]
        else:
            cur.append(w)
    cells.append(cur)
    return [(" ".join(w.text for w in c), c[0].x0, max(w.x1 for w in c)) for c in cells]


def role_of(text: str) -> str | None:
    t = re.sub(r"[^a-z/ ]", " ", text.lower())
    t = re.sub(r"\s*/\s*", "/", re.sub(r"\s+", " ", t)).strip()
    if not t or len(t) > 32:
        return None
    for role, rx in ROLE_RULES:
        if rx.search(t):
            return role
    return None


def _has_date_or_amount(line: Line) -> bool:
    return any(looks_like_amount(w.text) or parse_date(w.text) for w in line.words)


# ------------------------------------------------------------------ header detection
def find_header(lines: list[Line], cw: float):
    """Return (Header, index_of_last_header_line) or (None, -1)."""
    for i, ln in enumerate(lines):
        if _has_date_or_amount(ln) or len(ln.words) > 25:
            continue
        cells = split_cells(ln, gap=1.3 * cw)
        roles = {role_of(c[0]) for c in cells} - {None}
        if not (roles & {"date", "value_date"}) or len(roles) < 3 or not (roles & NUMERIC_ROLES):
            continue
        # merge an adjacent header line (e.g. "Txn" above "Date", or "Amt." below "Withdrawal")
        h = ln.bottom - ln.top
        texts = [[c[0]] for c in cells]
        last = i
        for j in (i - 1, i + 1):
            if not 0 <= j < len(lines):
                continue
            nb = lines[j]
            if abs(nb.top - ln.top) > 2.0 * h or _has_date_or_amount(nb):
                continue
            ncells = split_cells(nb, gap=1.3 * cw)
            hits = 0
            for (t, a, b) in ncells:
                k = _best_overlap(a, b, [(c[1], c[2]) for c in cells])
                if k is not None:
                    hits += 1
            if hits != len(ncells):
                continue            # not a header continuation (some cell matches nothing)
            for (t, a, b) in ncells:
                k = _best_overlap(a, b, [(c[1], c[2]) for c in cells])
                if j < i:
                    texts[k].insert(0, t)
                else:
                    texts[k].append(t)
            last = max(last, j)
        hcells = []
        for (t, a, b), parts in zip(cells, texts):
            full = " ".join(parts)
            hcells.append(HeaderCell(role_of(full) or "ignore", a, b, full))
        _dedupe_roles(hcells)
        top = min(lines[k].top for k in range(max(0, i - 1), last + 1) if abs(lines[k].top - ln.top) <= 2 * h)
        return Header(hcells, top, lines[last].bottom), last
    return None, -1


def _dedupe_roles(cells: list[HeaderCell]):
    seen = set()
    for c in cells:
        if c.role == "date" and "date" in seen and "value_date" not in seen:
            c.role = "value_date"
        if c.role in seen and c.role != "ignore":
            c.role = "ignore"
        seen.add(c.role)


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _best_overlap(a, b, spans):
    best, k = 0.0, None
    for idx, (s0, s1) in enumerate(spans):
        ov = _overlap(a, b, s0, s1)
        if ov > best:
            best, k = ov, idx
    return k


# ------------------------------------------------------------------ column boundaries
def build_columns(header: Header, body: list[Line], cw: float) -> list[Column]:
    rows = [l for l in body if any(looks_like_amount(w.text) for w in l.words)
            and any(parse_date(w.text) for w in l.words[:4])]
    cells = sorted(header.cells, key=lambda c: c.x0)
    spans = {i: None for i in range(len(cells))}

    if rows:
        res = 0.5
        width = max(l.x1 for l in rows) + 5
        cov = np.zeros(int(width / res) + 2)
        for l in rows:
            occ = np.zeros_like(cov, dtype=bool)
            for w in l.words:
                occ[int(w.x0 / res): int(w.x1 / res) + 1] = True
            cov += occ
        thr = int(0.04 * len(rows))
        filled = cov > thr
        segs, start, gap_run = [], None, 0
        min_gap = max(2.5, 0.9 * cw) / res
        for i, f in enumerate(filled):
            if f:
                if start is None:
                    start = i
                elif gap_run >= min_gap:
                    segs.append((start, last_filled))
                    start = i
                gap_run = 0
                last_filled = i
            elif start is not None:
                gap_run += 1
        if start is not None:
            segs.append((start, last_filled))
        segs = [(a * res, b * res) for a, b in segs]
        # map each whitespace-separated band to the header cell it belongs to
        for a, b in segs:
            k = _best_overlap(a, b, [(c.x0, c.x1) for c in cells])
            if k is None:  # no overlap -> nearest header cell
                k = min(range(len(cells)), key=lambda i: min(abs(a - cells[i].x1), abs(b - cells[i].x0)))
            cur = spans[k]
            spans[k] = (a, b) if cur is None else (min(cur[0], a), max(cur[1], b))

    # header cells that received no band: either an empty column or merged into a neighbour's band
    for i, c in enumerate(cells):
        if spans[i] is not None:
            continue
        spans[i] = (c.x0, c.x1)
        for j, sp in spans.items():
            if j == i or sp is None or _overlap(sp[0], sp[1], c.x0, c.x1) <= 0:
                continue
            left, right = (j, i) if cells[j].x0 < c.x0 else (i, j)
            if cells[right].role in NUMERIC_ROLES:
                split = max(cells[left].x1 + cw, cells[right].x1 - 13 * cw)
            else:
                split = cells[right].x0 - 0.5 * cw
            if j == left:
                spans[j] = (sp[0], min(sp[1], split))
                spans[i] = (split, max(c.x1, sp[1]))
            else:
                spans[j] = (max(sp[0], split), sp[1])
                spans[i] = (min(c.x0, sp[0]), split)
            break

    order = sorted(range(len(cells)), key=lambda i: spans[i][0])
    cols = []
    for pos, i in enumerate(order):
        left = -1e9 if pos == 0 else (spans[order[pos - 1]][1] + spans[i][0]) / 2
        right = 1e9 if pos == len(order) - 1 else (spans[i][1] + spans[order[pos + 1]][0]) / 2
        cols.append(Column(cells[i].role, left, right))
    return cols


def assign(line: Line, cols: list[Column]) -> dict:
    out: dict[str, list[Word]] = {}
    for w in line.words:
        for c in cols:
            if c.left <= w.cx < c.right:
                out.setdefault(c.role, []).append(w)
                break
    return out


# ------------------------------------------------------------------ row assembly
def _txt(cm, role):
    return " ".join(w.text for w in cm.get(role, []))


def join_fragments(frags: list[tuple[str, float]], right_edge: float, cw: float) -> str:
    """Join wrapped narration lines. A line that ends at the column edge with no trailing space was
    broken mid-word (common for long UPI strings) -> join without a space."""
    out = ""
    for k, (t, x1) in enumerate(frags):
        if not t:
            continue
        if not out:
            out = t
            continue
        prev_x1 = frags[k - 1][1]
        last_word = out.split(" ")[-1]
        glued = right_edge - prev_x1 < 2.5 * cw and len(last_word) >= 6
        out = out + ("" if glued else " ") + t
    return re.sub(r"\s+", " ", out).strip()


DRCR_WORDS = {"dr": "dr", "d": "dr", "debit": "dr", "withdrawal": "dr", "cr": "cr", "c": "cr",
              "credit": "cr", "deposit": "cr"}


class TableParser:
    """Stateful across pages: remembers the header/columns for pages that repeat no header."""

    def __init__(self):
        self.header: Header | None = None
        self.transactions: list[Transaction] = []
        self.opening: float | None = None
        self.closing: float | None = None
        self.notes: list[str] = []
        self._frags: dict[int, list] = {}   # id(txn) -> description fragments
        self._desc_right = 0.0
        self._cw = 4.0

    # -------------------------------------------------- public
    def feed_page(self, page_no: int, lines: list[Line], cw: float) -> tuple[int, bool]:
        self._cw = cw
        header, last_idx = find_header(lines, cw)
        if header:
            self.header = header
            body = lines[last_idx + 1:]
        elif self.header:
            body = lines
        else:
            return 0, False
        cols = build_columns(self.header, body, cw)
        self._cols = cols
        desc_cols = [c for c in cols if c.role == "description"]
        before = len(self.transactions)
        self._parse_body(page_no, body, cols)
        new = self.transactions[before:]
        if desc_cols:
            xs = [x1 for t in new for (_, x1) in self._frags.get(id(t), [])]
            self._desc_right = max(xs) if xs else self._desc_right
        for t in new:
            t.description = join_fragments(self._frags.pop(id(t), []), self._desc_right, cw) or t.description
        return len(new), header is not None

    # -------------------------------------------------- internals
    def _parse_body(self, page_no, body, cols):
        tops = [l.top for l in body]
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0.5]
        pitch = statistics.median(diffs) if diffs else 10.0
        current: Transaction | None = None
        cur_bottom = None
        pending: list[Line] = []
        prev_line_bottom = None

        for ln in body:
            cm = assign(ln, cols)
            date_txt = _txt(cm, "date")
            d = parse_date(date_txt) or (find_date_prefix(date_txt)[0] if date_txt else None)
            desc_txt = _txt(cm, "description")
            has_numbers = any(looks_like_amount(_txt(cm, r)) or parse_amount(_txt(cm, r))
                              for r in NUMERIC_ROLES if cm.get(r))

            if d:
                if SPECIAL_RE.search(desc_txt or ln.text) and not any(cm.get(r) for r in ("debit", "credit", "amount")):
                    self._special(ln.text, cm)
                    pending, prev_line_bottom = [], ln.bottom
                    continue
                txn = self._make_txn(d, cm, page_no)
                if txn is None:
                    continue
                self._distribute(pending, current, cur_bottom, ln, txn, pitch)
                pending = []
                self._frags.setdefault(id(txn), []).append((desc_txt, max((w.x1 for w in cm.get("description", [])), default=0)))
                self.transactions.append(txn)
                current, cur_bottom = txn, ln.bottom
                prev_line_bottom = ln.bottom
                continue

            if STOP_RE.match(ln.text):
                m = re.search(r"closing\s+bal", ln.text, re.I)
                if m:
                    self._special(ln.text, cm)
                break
            if SPECIAL_RE.search(ln.text):
                self._special(ln.text, cm)
                prev_line_bottom = ln.bottom
                continue

            only_text_cols = all(r in ("description", "ref", "ignore") for r in cm)
            close = prev_line_bottom is None or (ln.top - prev_line_bottom) <= 2.5 * pitch
            if desc_txt and only_text_cols and close:
                pending.append(ln)
                prev_line_bottom = ln.bottom
            elif has_numbers and current is not None and desc_txt and not cm.get("date"):
                # a row whose date was lost (e.g. OCR miss): keep it, inherit the previous date, flag it
                txn = self._make_txn(current.date, cm, page_no)
                if txn:
                    txn.flags.append("date_inferred")
                    self._distribute(pending, current, cur_bottom, ln, txn, pitch)
                    pending = []
                    self._frags.setdefault(id(txn), []).append((desc_txt, max(w.x1 for w in cm["description"])))
                    self.transactions.append(txn)
                    current, cur_bottom = txn, ln.bottom
                    prev_line_bottom = ln.bottom
        # leftover continuation lines belong to the last row
        if pending and current is not None:
            for p in pending:
                self._add_frag(current, p, cols)

    def _add_frag(self, txn, line, cols):
        cm = assign(line, cols)
        t = _txt(cm, "description")
        if t:
            self._frags.setdefault(id(txn), []).append((t, max(w.x1 for w in cm["description"])))
        r = _txt(cm, "ref")
        if r and not txn.ref:
            txn.ref = r

    def _distribute(self, pending, prev_txn, prev_bottom, next_line, next_txn, pitch):
        """Wrapped narration lines between two dated rows: split them at the biggest vertical gap.
        Lines above the gap belong to the previous row, lines below it to the next row
        (handles banks that vertically centre the date within a multi-line row)."""
        if not pending:
            return
        cols = self._cols
        if prev_txn is None:
            seq_above = []
            below = [p for p in pending if next_line.top - p.bottom <= 1.5 * pitch * len(pending)]
        else:
            edges = [prev_bottom] + [p.top for p in pending] + [next_line.top]
            bottoms = [prev_bottom] + [p.bottom for p in pending]
            gaps = [edges[k + 1] - bottoms[k] for k in range(len(pending) + 1)]
            k_star = max(range(len(gaps)), key=lambda k: (round(gaps[k], 1), k))
            seq_above, below = pending[:k_star], pending[k_star:]
        for p in seq_above:
            self._add_frag(prev_txn, p, cols)
        if below:
            pre = []
            for p in below:
                cm = assign(p, cols)
                t = _txt(cm, "description")
                if t:
                    pre.append((t, max(w.x1 for w in cm["description"])))
            self._frags[id(next_txn)] = pre + self._frags.get(id(next_txn), [])

    def _make_txn(self, d, cm, page_no) -> Transaction | None:
        def amt(role):
            return parse_amount(_txt(cm, role)) if cm.get(role) else None

        debit, credit, amount, bal = amt("debit"), amt("credit"), amt("amount"), amt("balance")
        t = Transaction(date=d, description=_txt(cm, "description"), page=page_no)
        if debit or credit:
            t.debit = abs(debit.value) if debit else 0.0
            t.credit = abs(credit.value) if credit else 0.0
        elif amount:
            hint = amount.hint or DRCR_WORDS.get(_txt(cm, "drcr").lower().strip(". "))
            if hint == "dr" or amount.value < 0:
                t.debit = abs(amount.value)
            elif hint == "cr":
                t.credit = abs(amount.value)
            else:
                t.debit = abs(amount.value)
                t.direction_unknown = True
        if bal:
            t.balance = -abs(bal.value) if bal.hint == "dr" else bal.value
        if not (t.debit or t.credit or t.balance is not None):
            return None          # a dated line with no money in it is not a transaction
        vd = _txt(cm, "value_date")
        t.value_date = parse_date(vd) if vd else None
        t.ref = _txt(cm, "ref")
        return t

    def _special(self, text, cm):
        amounts = [w.text for r in NUMERIC_ROLES for w in cm.get(r, [])]
        val = None
        if cm.get("balance"):
            a = parse_amount(_txt(cm, "balance"))
            val = (-abs(a.value) if a.hint == "dr" else a.value) if a else None
        if val is None:
            nums = [parse_amount(m.group()) for m in re.finditer(r"[\d,]+\.\d{2}(\s?(cr|dr))?", text, re.I)]
            nums = [n for n in nums if n]
            if nums:
                val = -abs(nums[-1].value) if nums[-1].hint == "dr" else nums[-1].value
        if val is None:
            return
        low = text.lower()
        if re.search(r"opening|b\s*/\s*f|brought", low) and not self.transactions and self.opening is None:
            self.opening = val
        elif re.search(r"closing", low):
            self.closing = val


def parse_with_header(pages_lines: list[tuple[int, list[Line], float]]):
    """pages_lines: [(page_no, lines, char_width)]. Returns (parser, per_page_counts, header_flags)."""
    p = TableParser()
    counts, flags = {}, {}
    for page_no, lines, cw in pages_lines:
        n, found = p.feed_page(page_no, lines, cw)
        counts[page_no], flags[page_no] = n, found
    return p, counts, flags


# ------------------------------------------------------------------ fallback (no header found)
def parse_fallback(pages_lines):
    """Header-less parser: 'DATE [VALUE DATE] DESCRIPTION ... [AMOUNT] BALANCE' per line."""
    txns, opening = [], None
    counts = {}
    for page_no, lines, cw in pages_lines:
        tops = [l.top for l in lines]
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0.5]
        pitch = statistics.median(diffs) if diffs else 10.0
        current, prev_bottom, n = None, None, 0
        for ln in lines:
            words = ln.words[:]
            if words and re.fullmatch(r"\d{1,4}", words[0].text) and len(words) > 3:
                words = words[1:]                     # serial number column
            d, k = find_date_prefix(" ".join(w.text for w in words[:3]))
            if d:
                rest = words[k:]
                if rest:
                    d2, k2 = find_date_prefix(" ".join(w.text for w in rest[:3]))
                    if d2:
                        rest = rest[k2:]              # value date
                nums, i = [], len(rest) - 1
                while i >= 0:
                    tx = rest[i].text
                    if tx.lower() in ("cr", "dr", "cr.", "dr.") and i > 0 and looks_like_amount(rest[i - 1].text):
                        nums.insert(0, rest[i - 1].text + " " + tx)
                        i -= 2
                    elif looks_like_amount(tx):
                        nums.insert(0, tx)
                        i -= 1
                    else:
                        break
                desc = " ".join(w.text for w in rest[: i + 1])
                if not nums:
                    continue
                if SPECIAL_RE.search(desc):
                    a = parse_amount(nums[-1])
                    if a and opening is None and not txns:
                        opening = -abs(a.value) if a.hint == "dr" else a.value
                    continue
                t = Transaction(date=d, description=desc, page=page_no)
                bal = parse_amount(nums[-1]) if len(nums) >= 2 else None
                amt = parse_amount(nums[-2] if len(nums) >= 2 else nums[-1])
                if bal:
                    t.balance = -abs(bal.value) if bal.hint == "dr" else bal.value
                if amt:
                    if amt.hint == "cr":
                        t.credit = abs(amt.value)
                    elif amt.hint == "dr" or amt.value < 0:
                        t.debit = abs(amt.value)
                    else:
                        t.debit, t.direction_unknown = abs(amt.value), True
                t.flags.append("fallback_parser")
                txns.append(t)
                current, prev_bottom, n = t, ln.bottom, n + 1
            elif current is not None and prev_bottom is not None and ln.top - prev_bottom <= 1.6 * pitch \
                    and not any(looks_like_amount(w.text) for w in ln.words) and not STOP_RE.match(ln.text):
                current.description += " " + ln.text
                prev_bottom = ln.bottom
            elif STOP_RE.match(ln.text):
                current = None
        counts[page_no] = n
    return txns, opening, counts
