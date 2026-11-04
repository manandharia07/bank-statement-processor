"""
End-to-end pipeline:  PDF bytes -> StatementResult (account details + validated, classified transactions).
"""
from __future__ import annotations

from datetime import date

from .account import extract_account
from .layout import char_width, group_lines, parse_fallback, parse_with_header
from .loader import NotAStatement, extract_pages, open_pdf
from .models import StatementResult
from .validator import _chain_score, check_dates, choose_order, reconcile, remove_duplicates, resolve_directions

LOW_OCR_CONFIDENCE = 60


def process_pdf(data: bytes, filename: str = "statement.pdf", password: str | None = None,
                force_ocr: bool = False, dpi: int = 300, classifier=None, progress=None) -> StatementResult:
    doc = open_pdf(data, password)
    pages, pages_lines = [], []
    for info, words in extract_pages(doc, force_ocr=force_ocr, dpi=dpi, progress=progress):
        pages.append(info)
        pages_lines.append((info.number, group_lines(words), char_width(words)))
    if not any(lines for _, lines, _ in pages_lines):
        raise NotAStatement("No readable text was found in this PDF, even after OCR.")

    # Two independent parsers; keep whichever result reconciles better.
    parser, counts, header_flags = parse_with_header(pages_lines)
    fb_txns, fb_open, fb_counts = parse_fallback(pages_lines)
    head_txns, head_open = parser.transactions, parser.opening

    account = extract_account(pages_lines)
    stated_open = account.get("opening_balance")

    def quality(txns, opening):
        if not txns:
            return (-1, 0)
        ordered, _ = choose_order(txns, opening if opening is not None else stated_open)
        return (_chain_score(ordered, opening if opening is not None else stated_open), len(txns))

    if quality(head_txns, head_open) >= quality(fb_txns, fb_open):
        txns, opening, used = head_txns, head_open, "header-mapped"
    else:
        txns, opening, used, counts = fb_txns, fb_open, "fallback", fb_counts
    if not txns:
        raise NotAStatement("No transaction table was found. Is this a bank statement? "
                            "(Looked for a header with Date / Description / Debit / Credit / Balance columns.)")

    if opening is None:
        opening = stated_open
    warnings = []
    txns, dupes = remove_duplicates(txns)
    if dupes:
        warnings.append(f"{dupes} duplicate row(s) removed (repeated page or overlapping data).")
    txns, reversed_ = choose_order(txns, opening)
    pf, pt = account.get("period_from"), account.get("period_to")
    fixed = check_dates(txns, date.fromisoformat(pf) if pf else None, date.fromisoformat(pt) if pt else None)
    if fixed:
        warnings.append(f"{fixed} date(s) looked misread (outside period / out of order) - corrected or flagged.")
    resolve_directions(txns, opening)
    closing_stated = account.get("closing_balance") or parser.closing
    recon = reconcile(txns, opening, closing_stated)
    warnings.extend(recon.pop("notes"))

    for p in pages:
        p.transactions = counts.get(p.number, 0)
        p.header_found = bool(header_flags.get(p.number))
        if p.kind == "ocr" and p.ocr_confidence is not None and p.ocr_confidence < LOW_OCR_CONFIDENCE:
            warnings.append(f"Page {p.number}: low OCR confidence ({p.ocr_confidence}%) - please review its rows.")
    mism = sum(1 for t in txns if any(f.startswith("balance_mismatch") for f in t.flags))
    if mism:
        warnings.append(f"{mism} row(s) do not reconcile with the running balance - highlighted for review.")
    unknown = sum(1 for t in txns if "direction_unknown" in t.flags)
    if unknown:
        warnings.append(f"{unknown} row(s): could not tell debit from credit (no Dr/Cr marker, no balance).")
    if recon.get("closing_matches") is False:
        warnings.append("Closing balance printed on the statement differs from the last row's balance.")

    if classifier is not None:
        classifier.classify_all(txns)

    return StatementResult(
        filename=filename, account=account, transactions=txns, pages=pages, warnings=warnings,
        reconciliation=recon, parser=used,
        order="newest-first (re-ordered to chronological)" if reversed_ else "chronological",
    )
