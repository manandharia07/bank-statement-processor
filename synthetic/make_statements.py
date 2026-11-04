"""
Generates realistic sample bank statements (fictional banks) for testing and demo.

Each layout deliberately exercises a different real-world variation:

  A  Navyug Bank        separate Withdrawal/Deposit columns, dd/mm/yy dates, Chq/Ref + Value Date
                        columns, char-wrapped UPI narrations, header repeated on every page.
  B  Sahyadri Sahakari  two-line table header, dd-Mon-yyyy dates, balance with "Cr" suffix,
                        NEWEST-FIRST ordering, opening balance only in the header block.
  C  Indus Valley Bank  single Amount column + separate Dr/Cr column, dd.mm.yyyy dates,
                        "Balance B/F" / "Balance C/F" rows, header ONLY on page 1.

Variants produced from those:
  *_scanned.pdf    every page rasterised, slightly rotated + noise (image-only PDF -> OCR)
  *_mixed.pdf      page 1 text, remaining pages scanned
  *_protected.pdf  password protected (password printed in README / samples/README.md)
  not_a_statement.pdf   to demonstrate graceful rejection

Ground truth CSVs are written to samples/ground_truth/ so extraction accuracy can be measured.

Run:  python -m synthetic.make_statements
"""
from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from synthetic.narrations import CATEGORY_BY_NAME, narration

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"
GT = OUT / "ground_truth"
PASSWORD = "RAHU1990"


# ------------------------------------------------------------------ transactions
@dataclass
class Txn:
    date: date
    desc: str
    debit: float
    credit: float
    balance: float = 0.0
    ref: str = ""
    category: str = ""


def _amt(rng, lo, hi, round_to=None):
    v = rng.uniform(lo, hi)
    if round_to:
        return float(round(v / round_to) * round_to)
    return round(v, 2)


def make_transactions(seed: int, start=date(2024, 4, 1), months=3, opening=45000.0):
    rng = random.Random(seed)
    events = []

    def add(d, cat, amount=None, text=None):
        c = CATEGORY_BY_NAME[cat]
        amount = amount if amount is not None else _amt(rng, *c.amount)
        text = text or narration(c, rng)
        if c.direction == "debit":
            events.append(Txn(d, text, amount, 0.0, category=cat))
        else:
            events.append(Txn(d, text, 0.0, amount, category=cat))

    salary = _amt(rng, 55000, 95000, 100)
    rent = _amt(rng, 12000, 25000, 500)
    emi = _amt(rng, 4000, 12000, 1)
    for mi in range(months):
        y, m = start.year + (start.month - 1 + mi) // 12, (start.month - 1 + mi) % 12 + 1
        day = lambda a, b: date(y, m, rng.randint(a, b))
        add(date(y, m, 1), "Salary", salary)
        add(day(2, 5), "Rent", rent)
        add(date(y, m, 5), "EMI & Loans", emi)
        add(day(8, 12), "Investments", _amt(rng, 2000, 10000, 500))
        add(day(14, 18), "Credit Card Payment")
        for _ in range(2):
            add(day(3, 25), "Utilities & Bills")
        weights = {"Food & Dining": 4, "Groceries": 3, "Shopping": 2, "Travel & Transport": 2, "Fuel": 1,
                   "Healthcare": 1, "Entertainment & Subscriptions": 2, "Transfer Out": 2, "Cash Withdrawal": 1,
                   "Education": 0.3, "Insurance": 0.3, "Taxes": 0.2}
        cats, w = zip(*weights.items())
        for _ in range(rng.randint(9, 13)):
            add(day(1, 28), rng.choices(cats, w)[0])
        add(day(6, 26), "Transfer In")
        if rng.random() < 0.6:
            add(day(6, 26), "Refunds & Cashback")
        if rng.random() < 0.5:
            add(day(6, 26), "Bank Charges")
        if rng.random() < 0.3:
            add(day(6, 26), "Cash Deposit")
        if mi == months - 1:
            add(date(y, m, 28), "Interest & Dividends")

    events.sort(key=lambda t: t.date)
    bal, out = opening, []
    for t in events:
        if t.debit and bal - t.debit < 1000:     # keep the account realistic (no overdraft)
            continue
        bal = round(bal + t.credit - t.debit, 2)
        t.balance = bal
        t.ref = rng.choice(["", "", f"{rng.randint(10**11, 10**12 - 1)}", f"0000{rng.randint(10**7, 10**8 - 1)}"])
        out.append(t)
    return out, opening


# ------------------------------------------------------------------ formatting helpers
def inr(v: float) -> str:
    """Indian digit grouping: 1234567.5 -> 12,34,567.50"""
    neg = v < 0
    s = f"{abs(v):.2f}"
    whole, frac = s.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return ("-" if neg else "") + whole + "." + frac


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    """Greedy word wrap; words wider than the column are broken by characters (like real banks do)."""
    lines, cur = [], ""
    for word in text.split(" "):
        cand = word if not cur else cur + " " + word
        if stringWidth(cand, font, size) <= width:
            cur = cand
            continue
        if cur:
            lines.append(cur)
            cur = ""
        while stringWidth(word, font, size) > width:
            k = len(word)
            while k > 1 and stringWidth(word[:k], font, size) > width:
                k -= 1
            lines.append(word[:k])
            word = word[k:]
        cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


@dataclass
class Col:
    key: str
    header: list
    x: float
    w: float
    align: str = "L"


@dataclass
class Layout:
    name: str
    bank: str
    cols: list
    datefmt: str
    font: str = "Helvetica"
    size: float = 7.5
    lead: float = 9.2
    header_every_page: bool = True
    newest_first: bool = False
    balance_suffix: bool = False
    bf_cf_rows: bool = False
    opening_row: bool = False
    info: list = field(default_factory=list)   # list of (left, right) strings
    summary: bool = True


HOLDER = "RAHUL VERMA"
LAYOUTS = {
    "A": Layout(
        name="navyug_bank", bank="NAVYUG BANK LTD", datefmt="%d/%m/%y",
        cols=[Col("date", ["Date"], 30, 38), Col("desc", ["Narration"], 72, 168),
              Col("ref", ["Chq./Ref.No."], 246, 60), Col("vdate", ["Value Dt"], 312, 38),
              Col("debit", ["Withdrawal Amt."], 356, 62, "R"), Col("credit", ["Deposit Amt."], 424, 62, "R"),
              Col("balance", ["Closing Balance"], 492, 72, "R")],
        opening_row=False,
        info=[("Account Holder Name : " + HOLDER, "Account No : 50100234567891"),
              ("Address : FLAT 12, SUNRISE APTS, ANDHERI EAST", "Branch : ANDHERI EAST, MUMBAI"),
              ("MUMBAI - 400069, MAHARASHTRA", "IFSC Code : NAVB0000234"),
              ("Customer ID : 87654321", "MICR : 400240012"),
              ("Account Type : SAVINGS", "Currency : INR"),
              ("Statement Period : From 01/04/2024 To 30/06/2024", "")],
    ),
    "B": Layout(
        name="sahyadri_bank", bank="SAHYADRI SAHAKARI BANK LIMITED", datefmt="%d-%b-%Y", size=8, lead=9.8,
        cols=[Col("date", ["Txn", "Date"], 32, 52), Col("vdate", ["Value", "Date"], 88, 52),
              Col("desc", ["Description"], 146, 190), Col("debit", ["Debit"], 342, 66, "R"),
              Col("credit", ["Credit"], 414, 66, "R"), Col("balance", ["Balance"], 486, 78, "R")],
        newest_first=True, balance_suffix=True,
        info=[("Name: Mr. " + HOLDER, "A/C No: 001234567890123"),
              ("IFSC: SAHY0000045", "Branch: KOTHRUD, PUNE"),
              ("Cust ID: SSB00451278", "Account Type: Savings Account"),
              ("Period: 01-Apr-2024 to 30-Jun-2024", "Opening Balance: {opening} Cr")],
    ),
    "C": Layout(
        name="indus_valley_bank", bank="INDUS VALLEY BANK", datefmt="%d.%m.%Y", size=7.8, lead=9.6,
        cols=[Col("date", ["Date"], 30, 46), Col("desc", ["Particulars"], 82, 200),
              Col("ref", ["Cheque No"], 288, 50), Col("amount", ["Amount (INR)"], 344, 70, "R"),
              Col("drcr", ["Dr/Cr"], 422, 26), Col("balance", ["Balance (INR)"], 454, 80, "R")],
        header_every_page=False, bf_cf_rows=True, opening_row=True,
        info=[("Customer Name : " + HOLDER, "Account Number : 3456789012"),
              ("IFSC : INDV0001122", "MICR Code : 411256003"),
              ("Branch Name : FC ROAD, PUNE", "Statement From 01.04.2024 To 30.06.2024")],
    ),
}


def render(layout: Layout, txns: list[Txn], opening: float, path: Path):
    W, H = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(f"{layout.bank} - Statement of Account")
    f, fb, s, lead = layout.font, layout.font + "-Bold", layout.size, layout.lead
    rows = list(reversed(txns)) if layout.newest_first else list(txns)

    def fmt_bal(v):
        return inr(v) + (" Cr" if layout.balance_suffix else "")

    def cells(t: Txn):
        d = {"date": t.date.strftime(layout.datefmt), "vdate": t.date.strftime(layout.datefmt),
             "desc": t.desc, "ref": t.ref,
             "debit": inr(t.debit) if t.debit else "", "credit": inr(t.credit) if t.credit else "",
             "amount": inr(t.debit or t.credit), "drcr": "Dr" if t.debit else "Cr",
             "balance": fmt_bal(t.balance)}
        if layout.name == "indus_valley_bank":
            d["ref"] = t.ref[-6:] if t.ref.startswith("0000") else ""
        return d

    def page_header(first: bool) -> float:
        y = H - 40
        c.setFont(fb, 14)
        c.drawString(30, y, layout.bank)
        c.setFont(f, 9)
        c.drawRightString(W - 30, y, "Statement of Account")
        y -= 22
        if first:
            c.setFont(f, 8.5)
            for left, right in layout.info:
                c.drawString(30, y, left.format(opening=inr(opening)))
                if right:
                    c.drawString(320, y, right.format(opening=inr(opening)))
                y -= 12
            y -= 8
        return y

    def table_header(y: float) -> float:
        nlines = max(len(col.header) for col in layout.cols)
        top = y + s + 2
        c.setFillGray(0.88)
        c.rect(28, top - nlines * lead - 4, W - 58, nlines * lead + 4, fill=1, stroke=0)
        c.setFillGray(0)
        c.setFont(fb, s)
        for col in layout.cols:
            for i, line in enumerate(col.header):
                yy = y - i * lead
                if col.align == "R":
                    c.drawRightString(col.x + col.w, yy, line)
                else:
                    c.drawString(col.x, yy, line)
        return y - nlines * lead - 6

    def simple_row(y, label, balance, d=None):
        c.setFont(f, s)
        if d:
            c.drawString(layout.cols[0].x, y, d.strftime(layout.datefmt))
        desc_col = next(col for col in layout.cols if col.key == "desc")
        bal_col = next(col for col in layout.cols if col.key == "balance")
        c.drawString(desc_col.x, y, label)
        c.drawRightString(bal_col.x + bal_col.w, y, fmt_bal(balance))
        return y - lead - 3

    def footer(pno):
        c.setFont(f, 7)
        c.drawCentredString(W / 2, 30, f"Page {pno}")
        c.drawString(30, 42, "This is a computer generated statement and does not require a signature.")

    pno, y = 1, page_header(True)
    y = table_header(y)
    bal_before = opening
    if layout.opening_row:
        y = simple_row(y, "Opening Balance", opening, txns[0].date.replace(day=1))
    for t in rows:
        cl = cells(t)
        desc_col = next(col for col in layout.cols if col.key == "desc")
        dlines = wrap(cl["desc"], f, s, desc_col.w)
        need = len(dlines) * lead + 3
        if y - need < 70:
            if layout.bf_cf_rows:
                simple_row(y, "Balance C/F", bal_before)
            footer(pno)
            c.showPage()
            pno += 1
            y = page_header(False)
            if layout.header_every_page:
                y = table_header(y)
            if layout.bf_cf_rows:
                y = simple_row(y, "Balance B/F", bal_before)
        c.setFont(f, s)
        for col in layout.cols:
            if col.key == "desc":
                for i, ln in enumerate(dlines):
                    c.drawString(col.x, y - i * lead, ln)
                continue
            v = cl.get(col.key, "")
            if not v:
                continue
            if col.align == "R":
                c.drawRightString(col.x + col.w, y, v)
            else:
                c.drawString(col.x, y, v)
        y -= need
        bal_before = t.balance

    if layout.summary:
        y -= 10
        if y < 120:
            footer(pno)
            c.showPage()
            pno += 1
            y = page_header(False) - 10
        tot_dr = sum(t.debit for t in txns)
        tot_cr = sum(t.credit for t in txns)
        c.setFont(fb, 8.5)
        c.drawString(30, y, "STATEMENT SUMMARY")
        c.setFont(f, 8.5)
        for i, (k, v) in enumerate([("Opening Balance", opening), ("Total Debits", tot_dr),
                                     ("Total Credits", tot_cr), ("Closing Balance", txns[-1].balance)]):
            c.drawString(30, y - 13 * (i + 1), f"{k} : {inr(v)}")
        c.drawString(320, y - 13, f"Debit Count : {sum(1 for t in txns if t.debit)}")
        c.drawString(320, y - 26, f"Credit Count : {sum(1 for t in txns if t.credit)}")
    footer(pno)
    c.save()


# ------------------------------------------------------------------ variants
def _scan_page(pix_img: Image.Image, rng: random.Random, angle: float) -> Image.Image:
    img = pix_img.convert("L").rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.default_rng(rng.randint(0, 9999)).normal(0, 9, arr.shape)   # sensor noise
    arr = np.clip(arr * 0.95 + 8, 0, 255).astype(np.uint8)                       # slight grey paper
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.4))


def make_scanned(src: Path, dst: Path, seed: int, pages_to_scan=None, dpi=200):
    rng = random.Random(seed)
    doc = fitz.open(src)
    out = fitz.open()
    for i, page in enumerate(doc):
        if pages_to_scan is not None and i not in pages_to_scan:
            out.insert_pdf(doc, from_page=i, to_page=i)
            continue
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = _scan_page(img, rng, angle=rng.uniform(-0.8, 0.8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=72)
        np_page = out.new_page(width=page.rect.width, height=page.rect.height)
        np_page.insert_image(np_page.rect, stream=buf.getvalue())
    out.save(dst, deflate=True)


def make_protected(src: Path, dst: Path, password: str):
    doc = fitz.open(src)
    doc.save(dst, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password + "_owner")


def make_not_statement(dst: Path):
    c = canvas.Canvas(str(dst), pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, 780, "Quarterly Team Newsletter")
    c.setFont("Helvetica", 11)
    for i, line in enumerate(["This document is not a bank statement.",
                              "It is used to verify that the system rejects unrelated PDFs gracefully",
                              "instead of crashing or producing garbage output."]):
        c.drawString(60, 740 - 18 * i, line)
    c.save()


def write_ground_truth(txns, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "description", "debit", "credit", "balance", "category"])
        for t in txns:
            w.writerow([t.date.isoformat(), t.desc, f"{t.debit:.2f}", f"{t.credit:.2f}", f"{t.balance:.2f}", t.category])


def main():
    OUT.mkdir(exist_ok=True)
    GT.mkdir(exist_ok=True)
    seeds = {"A": 11, "B": 22, "C": 33}
    files = {}
    for key, layout in LAYOUTS.items():
        txns, opening = make_transactions(seeds[key])
        p = OUT / f"{layout.name}.pdf"
        render(layout, txns, opening, p)
        write_ground_truth(txns, GT / f"{layout.name}.csv")
        files[key] = p
        print(f"wrote {p.name}: {len(txns)} transactions")

    make_scanned(files["A"], OUT / "navyug_bank_scanned.pdf", seed=1)
    make_scanned(files["C"], OUT / "indus_valley_bank_mixed.pdf", seed=2, pages_to_scan={1, 2, 3, 4})
    make_protected(files["B"], OUT / "sahyadri_bank_protected.pdf", PASSWORD)
    make_not_statement(OUT / "not_a_statement.pdf")
    for name, src in [("navyug_bank_scanned", "navyug_bank"), ("indus_valley_bank_mixed", "indus_valley_bank"),
                      ("sahyadri_bank_protected", "sahyadri_bank")]:
        (GT / f"{name}.csv").write_bytes((GT / f"{src}.csv").read_bytes())
    print("variants written; protected PDF password:", PASSWORD)


if __name__ == "__main__":
    main()
