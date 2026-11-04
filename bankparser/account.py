"""
Account-holder details (name, account number, IFSC, MICR, branch, period, balances...).

Statements print these as "Label : Value" pairs, often two per line. Each line is rebuilt
with " | " wherever there is a big horizontal gap, so a value never runs into the next label.
"""
from __future__ import annotations

import re

from .layout import split_cells
from .normalize import parse_amount, parse_date

IFSC_BANKS = {
    "HDFC": "HDFC Bank", "ICIC": "ICICI Bank", "SBIN": "State Bank of India", "UTIB": "Axis Bank",
    "KKBK": "Kotak Mahindra Bank", "PUNB": "Punjab National Bank", "BARB": "Bank of Baroda",
    "CNRB": "Canara Bank", "UBIN": "Union Bank of India", "IDIB": "Indian Bank", "YESB": "Yes Bank",
    "INDB": "IndusInd Bank", "IDFB": "IDFC FIRST Bank", "FDRL": "Federal Bank", "BKID": "Bank of India",
    "MAHB": "Bank of Maharashtra", "IOBA": "Indian Overseas Bank", "UCBA": "UCO Bank",
    "CBIN": "Central Bank of India", "PSIB": "Punjab & Sind Bank", "RATN": "RBL Bank",
    "AUBL": "AU Small Finance Bank", "KARB": "Karnataka Bank", "SIBL": "South Indian Bank",
    "CIUB": "City Union Bank", "KVBL": "Karur Vysya Bank", "DBSS": "DBS Bank India",
    "SCBL": "Standard Chartered", "HSBC": "HSBC", "CITI": "Citibank", "BDBL": "Bandhan Bank",
    "JAKA": "J&K Bank", "TMBL": "Tamilnad Mercantile Bank", "DCBL": "DCB Bank", "ESFB": "Equitas SFB",
}

_V = r"\s*[:\-]?\s*"          # optional ':' or '-' between label and value
PATTERNS = {
    "account_holder": [
        rf"(?:account\s*holder(?:'?s)?\s*name|customer\s*name|a/?c\s*name|account\s*name|name\s*of\s*(?:the\s*)?customer|^name){_V}"
        r"((?:mr|mrs|ms|m/s|shri|smt|dr)\.?\s+)?([A-Za-z][A-Za-z .']{2,60})",
    ],
    "account_number": [rf"(?:account|a/?c|acct)\.?\s*(?:no|number|num|#)\.?{_V}([0-9Xx*]{{6,20}})"],
    "ifsc": [rf"ifsc?\s*(?:code)?{_V}([A-Z]{{4}}[0O][A-Z0-9]{{6}})"],
    "micr": [rf"micr\s*(?:code|no)?{_V}(\d{{9}})"],
    "customer_id": [rf"(?:customer\s*id|cust\.?\s*id|cif\s*(?:no|id)?|client\s*id|crn){_V}([A-Z0-9]{{5,20}})"],
    "branch": [rf"branch\s*(?:name)?{_V}([A-Za-z0-9 ,.\-&()]{{3,60}})"],
    "account_type": [rf"(?:account\s*type|a/?c\s*type|type\s*of\s*account|product\s*name){_V}([A-Za-z ]{{3,40}})"],
    "currency": [rf"currency{_V}([A-Z]{{3}})"],
}
_DATE_PIECE = r"(\d{1,2}[/\-. ](?:\d{1,2}|[A-Za-z]{3,9})[/\-. ,]*\d{2,4})"
PERIOD_RE = re.compile(_DATE_PIECE + r"\s*(?:to|till|-|–)\s*" + _DATE_PIECE, re.I)
BAL_RE = {
    "opening_balance": re.compile(r"opening\s*bal(?:ance)?" + _V + r"(?:rs\.?|inr|₹)?\s*([\d,]+\.\d{2}\s*(?:cr|dr)?)", re.I),
    "closing_balance": re.compile(r"closing\s*bal(?:ance)?" + _V + r"(?:rs\.?|inr|₹)?\s*([\d,]+\.\d{2}\s*(?:cr|dr)?)", re.I),
}


def _cell_lines(lines, cw):
    return [" | ".join(c[0] for c in split_cells(l, gap=3.0 * cw)) for l in lines]


def extract_account(pages_lines) -> dict:
    """pages_lines: [(page_no, lines, char_width)] - looks at the first page (+ last page for summaries)."""
    if not pages_lines:
        return {}
    first = pages_lines[0]
    texts = _cell_lines(first[1][:60], first[2])
    tail = _cell_lines(pages_lines[-1][1], pages_lines[-1][2]) if len(pages_lines) > 1 else []
    info: dict = {}

    for field, pats in PATTERNS.items():
        for txt in texts:
            for part in txt.split(" | "):
                m = None
                for p in pats:
                    m = re.search(p, part.strip(), re.I)
                    if m:
                        break
                if m:
                    val = m.groups()[-1].strip(" .,:-")
                    if field == "ifsc":
                        val = val.upper()
                        val = val[:4] + "0" + val[5:]          # OCR: 5th char is always zero
                    if field == "account_holder" and re.search(r"\b(bank|branch|address)\b", val, re.I):
                        continue
                    info.setdefault(field, val)
                    break
            if field in info:
                break

    # IFSC without a label anywhere on the page
    if "ifsc" not in info:
        for txt in texts:
            m = re.search(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", txt)
            if m:
                info["ifsc"] = m.group(1)
                break

    # account holder fallback: a line starting with Mr/Mrs/Ms
    if "account_holder" not in info:
        for txt in texts:
            m = re.match(r"^(?:mr|mrs|ms|shri|smt)\.?\s+([A-Za-z .]{3,50})", txt, re.I)
            if m:
                info["account_holder"] = m.group(1).split(" | ")[0].strip()
                break

    for txt in texts:
        m = PERIOD_RE.search(txt.replace(" | ", " "))
        if m and re.search(r"period|from|statement|date", txt, re.I):
            d1, d2 = parse_date(m.group(1).strip(" ,")), parse_date(m.group(2).strip(" ,"))
            if d1 and d2:
                info["period_from"], info["period_to"] = d1.isoformat(), d2.isoformat()
                break

    for key, rx in BAL_RE.items():
        for txt in texts + tail:
            m = rx.search(txt.replace(" | ", " "))
            if m:
                a = parse_amount(m.group(1))
                if a:
                    info[key] = -abs(a.value) if a.hint == "dr" else a.value
                    break

    # bank name: IFSC prefix lookup, else a prominent line containing "bank"
    if info.get("ifsc", "")[:4] in IFSC_BANKS:
        info["bank_name"] = IFSC_BANKS[info["ifsc"][:4]]
    else:
        for txt in texts[:8]:
            m = re.search(r"([A-Za-z&.\- ]*\b(?:bank|co-?operative|sahakari)\b[A-Za-z&.\- ]*)", txt.split(" | ")[0], re.I)
            if m and len(m.group(1)) < 60 and not re.search(r"statement|account", m.group(1), re.I):
                info["bank_name"] = m.group(1).strip()
                break

    order = ["bank_name", "account_holder", "account_number", "ifsc", "micr", "customer_id", "branch",
             "account_type", "currency", "period_from", "period_to", "opening_balance", "closing_balance"]
    return {k: info[k] for k in order if k in info}
