from datetime import date

import pytest

from bankparser.normalize import fmt_inr, looks_like_amount, parse_amount, parse_date


@pytest.mark.parametrize("text,expected", [
    ("01/04/2024", date(2024, 4, 1)),
    ("01/04/24", date(2024, 4, 1)),
    ("05-Apr-2024", date(2024, 4, 5)),
    ("05 Apr 2024", date(2024, 4, 5)),
    ("5-Sept-24", date(2024, 9, 5)),
    ("28.06.2024", date(2024, 6, 28)),
    ("2024-06-28", date(2024, 6, 28)),
    ("O1/O4/2O24", date(2024, 4, 1)),          # OCR read zeros as letter O
    ("31/02/2024", None),                      # impossible date
    ("Narration", None),
    ("123456789012", None),
])
def test_parse_date(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("text,value,hint", [
    ("1,23,456.78", 123456.78, None),          # Indian grouping
    ("123,456.78", 123456.78, None),           # western grouping
    ("11,593.17 Cr", 11593.17, "cr"),
    ("450.00Dr", 450.0, "dr"),
    ("(1,250.00)", -1250.0, None),
    ("-99.50", -99.5, None),
    ("Rs. 1,000.00", 1000.0, None),
    ("₹ 2,500.00", 2500.0, None),
    ("1.234.00", 1234.0, None),                # OCR read ',' as '.'
    ("1,234,00", 1234.0, None),                # OCR read '.' as ','
    ("1,0O0.00", 1000.0, None),                # OCR letter O
])
def test_parse_amount(text, value, hint):
    a = parse_amount(text)
    assert a is not None and a.value == pytest.approx(value) and a.hint == hint


@pytest.mark.parametrize("text", ["", "-", "NIL", "abc"])
def test_parse_amount_empty(text):
    assert parse_amount(text) is None


def test_looks_like_amount():
    assert looks_like_amount("1,18,100.00")
    assert looks_like_amount("4,088.22 Cr")
    assert not looks_like_amount("000052408663")    # reference number, no decimals
    assert not looks_like_amount("01/04/24")


def test_fmt_inr():
    assert fmt_inr(12345678.9) == "1,23,45,678.90"
    assert fmt_inr(999) == "999.00"
