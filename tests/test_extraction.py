import csv
from pathlib import Path

import pytest

from bankparser import InvalidPDF, NotAStatement, PasswordRequired, WrongPassword, process_pdf
from bankparser.ocr import tesseract_available

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
PASSWORD = "RAHU1990"
needs_ocr = pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")


def _check_against_truth(name, password=None, min_field_accuracy=1.0):
    res = process_pdf((SAMPLES / f"{name}.pdf").read_bytes(), name, password=password)
    with open(SAMPLES / "ground_truth" / f"{name}.csv", encoding="utf-8") as fh:
        gt = list(csv.DictReader(fh))
    assert len(res.transactions) == len(gt)
    ok = 0
    for g, t in zip(gt, res.transactions):
        ok += (t.date.isoformat() == g["date"] and abs(t.debit - float(g["debit"])) < 0.005
               and abs(t.credit - float(g["credit"])) < 0.005 and abs(t.balance - float(g["balance"])) < 0.005)
    assert ok / len(gt) >= min_field_accuracy
    return res


@pytest.mark.parametrize("name", ["navyug_bank", "sahyadri_bank", "indus_valley_bank"])
def test_text_pdfs_extract_exactly(name):
    res = _check_against_truth(name)
    assert res.reconciliation["match_rate"] == 100.0
    assert all(p.kind == "text" for p in res.pages)
    assert res.account["account_holder"] == "RAHUL VERMA"
    assert res.account["period_from"] == "2024-04-01"


def test_newest_first_statement_is_reordered():
    res = _check_against_truth("sahyadri_bank")
    assert "newest-first" in res.order


def test_account_details_layout_a():
    res = process_pdf((SAMPLES / "navyug_bank.pdf").read_bytes())
    a = res.account
    assert a["account_number"] == "50100234567891"
    assert a["ifsc"] == "NAVB0000234"
    assert a["micr"] == "400240012"
    assert a["branch"] == "ANDHERI EAST, MUMBAI"


def test_password_protected():
    data = (SAMPLES / "sahyadri_bank_protected.pdf").read_bytes()
    with pytest.raises(PasswordRequired):
        process_pdf(data)
    with pytest.raises(WrongPassword):
        process_pdf(data, password="wrong")
    _check_against_truth("sahyadri_bank_protected", password=PASSWORD)


def test_not_a_statement_is_rejected():
    with pytest.raises(NotAStatement):
        process_pdf((SAMPLES / "not_a_statement.pdf").read_bytes())


def test_garbage_bytes_rejected():
    with pytest.raises(InvalidPDF):
        process_pdf(b"this is not a pdf")
    with pytest.raises(InvalidPDF):
        process_pdf(b"%PDF-1.7 truncated garbage")


@needs_ocr
def test_scanned_pdf_ocr():
    res = _check_against_truth("navyug_bank_scanned", min_field_accuracy=0.9)
    assert all(p.kind == "ocr" for p in res.pages)


@needs_ocr
def test_mixed_pdf_detects_per_page():
    res = _check_against_truth("indus_valley_bank_mixed", min_field_accuracy=0.9)
    assert [p.kind for p in res.pages] == ["text", "ocr"]
