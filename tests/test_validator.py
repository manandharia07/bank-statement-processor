from datetime import date

from bankparser.models import Transaction
from bankparser.validator import choose_order, reconcile, remove_duplicates, resolve_directions


def T(day, debit=0.0, credit=0.0, balance=None, unknown=False, desc="X"):
    t = Transaction(date(2024, 4, day), desc, debit, credit, balance)
    t.direction_unknown = unknown
    return t


def test_reconcile_all_ok():
    tx = [T(1, credit=100, balance=1100), T(2, debit=50, balance=1050)]
    r = reconcile(tx, 1000.0)
    assert r["match_rate"] == 100.0 and r["closing_balance"] == 1050


def test_swapped_direction_is_corrected():
    tx = [T(1, debit=100, balance=1100)]              # really a credit
    reconcile(tx, 1000.0)
    assert tx[0].credit == 100 and "direction_corrected" in tx[0].flags


def test_mismatch_flagged():
    tx = [T(1, debit=100, balance=950)]               # OCR misread something
    r = reconcile(tx, 1000.0)
    assert r["rows_matched"] == 0 and tx[0].flags[0].startswith("balance_mismatch")


def test_direction_from_balance():
    tx = [T(1, debit=200, balance=1200, unknown=True), T(2, debit=50, balance=1150, unknown=True)]
    resolve_directions(tx, 1000.0)
    assert tx[0].credit == 200 and tx[1].debit == 50


def test_newest_first_detected():
    tx = [T(3, debit=50, balance=1050), T(2, credit=100, balance=1100)]
    ordered, rev = choose_order(tx, 1000.0)
    assert rev and ordered[0].date.day == 2


def test_duplicates_removed_but_real_repeats_kept():
    a = T(1, debit=10, balance=990, desc="TEA")
    b = T(1, debit=10, balance=980, desc="TEA")        # genuine 2nd purchase, different balance
    c = T(1, debit=10, balance=980, desc="TEA")        # repeated page
    out, removed = remove_duplicates([a, b, c])
    assert len(out) == 2 and removed == 1


def test_missing_opening_inferred():
    tx = [T(1, debit=100, balance=900), T(2, debit=100, balance=800)]
    r = reconcile(tx, None)
    assert r["opening_balance"] == 1000 and r["match_rate"] == 100.0
