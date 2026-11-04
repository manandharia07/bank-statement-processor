"""
Validation by balance reconciliation - the main accuracy check.

For every row:  previous_balance + credit - debit  must equal  balance.
This one rule lets the system
  * prove extraction accuracy (reconciliation rate),
  * detect statement order (newest-first statements reconcile only when reversed),
  * work out Dr/Cr when the statement has a single amount column without a Dr/Cr marker,
  * auto-correct swapped debit/credit and fill a missing amount or balance,
  * flag rows where OCR probably misread a number.
"""
from __future__ import annotations

TOL = 0.01


def _chain_score(txns, opening) -> int:
    prev, ok = opening, 0
    for t in txns:
        if t.balance is None:
            prev = None
            continue
        if prev is not None:
            delta = round(t.balance - prev, 2)
            amt = t.credit - t.debit
            if abs(delta - amt) <= TOL or (t.direction_unknown and abs(abs(delta) - t.debit) <= TOL):
                ok += 1
        prev = t.balance
    return ok


def choose_order(txns, opening):
    """Return (txns_in_chronological_order, was_reversed)."""
    if len(txns) < 2:
        return txns, False
    rev = list(reversed(txns))
    fwd_score, rev_score = _chain_score(txns, opening), _chain_score(rev, opening)
    dated = [t.date for t in txns if t.date]
    descending = len(dated) >= 2 and dated[0] > dated[-1]
    if rev_score > fwd_score or (rev_score == fwd_score and descending):
        return rev, True
    return txns, False


def remove_duplicates(txns):
    """Rows identical in date, description, amounts AND running balance can only be a repeated
    page / overlapping extract - genuine repeat payments always have a different balance."""
    seen, out, removed = set(), [], 0
    for t in txns:
        key = (t.date, " ".join(t.description.split()).upper(), round(t.debit, 2), round(t.credit, 2),
               None if t.balance is None else round(t.balance, 2))
        if key in seen and t.balance is not None:
            removed += 1
            continue
        seen.add(key)
        out.append(t)
    return out, removed


def resolve_directions(txns, opening):
    prev = opening
    for t in txns:
        if t.direction_unknown and t.balance is not None and prev is not None:
            delta = round(t.balance - prev, 2)
            if abs(delta - t.debit) <= TOL:
                t.credit, t.debit = t.debit, 0.0
                t.direction_unknown = False
                t.flags.append("direction_from_balance")
            elif abs(delta + t.debit) <= TOL:
                t.direction_unknown = False
                t.flags.append("direction_from_balance")
        if t.direction_unknown:
            t.flags.append("direction_unknown")
        prev = t.balance if t.balance is not None else None


def reconcile(txns, opening, closing_stated=None) -> dict:
    notes = []
    if opening is None and txns and txns[0].balance is not None:
        opening = round(txns[0].balance - txns[0].credit + txns[0].debit, 2)
        notes.append("Opening balance not printed - inferred from the first row.")
        first_unverified = True
    else:
        first_unverified = False

    prev, checked, matched = opening, 0, 0
    for i, t in enumerate(txns):
        if t.balance is None:
            if prev is not None:
                t.balance = round(prev + t.credit - t.debit, 2)
                t.flags.append("balance_computed")
            prev = t.balance
            continue
        if prev is None or (i == 0 and first_unverified):
            prev = t.balance
            continue
        checked += 1
        expected = round(prev + t.credit - t.debit, 2)
        if abs(expected - t.balance) <= TOL:
            matched += 1
        elif (t.debit or t.credit) and abs(prev - t.credit + t.debit - t.balance) <= TOL:
            t.debit, t.credit = t.credit, t.debit
            t.flags.append("direction_corrected")
            matched += 1
        elif not t.debit and not t.credit:
            delta = round(t.balance - prev, 2)
            if delta >= 0:
                t.credit = delta
            else:
                t.debit = -delta
            t.flags.append("amount_inferred")
            matched += 1
        else:
            t.flags.append(f"balance_mismatch({t.balance - expected:+.2f})")
        prev = t.balance

    closing = txns[-1].balance if txns else None
    summary = {
        "opening_balance": opening,
        "closing_balance": closing,
        "stated_closing_balance": closing_stated,
        "closing_matches": None if closing_stated is None or closing is None else abs(closing - closing_stated) <= TOL,
        "rows_checked": checked,
        "rows_matched": matched,
        "match_rate": round(100.0 * matched / checked, 1) if checked else None,
        "total_debit": round(sum(t.debit for t in txns), 2),
        "total_credit": round(sum(t.credit for t in txns), 2),
        "notes": notes,
    }
    return summary
