import pytest

from bankparser.classify.hybrid import HybridClassifier
from bankparser.classify.rules import RuleEngine
from bankparser.models import Transaction

rules = RuleEngine()


@pytest.mark.parametrize("text,direction,expected", [
    ("UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/UPI", "debit", "Food & Dining"),
    ("UPI/DR/412345678901/SWIGGY INSTAMART/YESB/x@ybl/UPI", "debit", "Groceries"),
    ("NEFT CR-HDFC0001234-ACME TECHNOLOGIES-SALARY APR 2024", "credit", "Salary"),
    ("REFUND/AMAZON/1234567890", "credit", "Refunds & Cashback"),
    ("POS 4123XXXXXX1234 AMAZON MUMBAI", "debit", "Shopping"),
    ("NWD-4123XXXXXX1234-S1AW1234-PUNE", "debit", "Cash Withdrawal"),
    ("TDS ON INTEREST", "debit", "Taxes"),
    ("CREDIT INTEREST", "credit", "Interest & Dividends"),
    ("BIL/BPAY/123456789/FASTAG RECHARGE/9876543210", "debit", "Travel & Transport"),  # not 'Bank Charges'
    ("UPI/DR/1234/zomato12@paytm/ICIC", "debit", "Food & Dining"),                       # joined token
])
def test_rules(text, direction, expected):
    assert rules.match(text, direction)[0] == expected


def test_rule_direction_constraint():
    # 'SALARY' as a DEBIT is not salary income
    assert rules.match("SALARY ADVANCE RECOVERY", "debit")[0] != "Salary"


def test_ml_fallback_and_direction_mask(tmp_path):
    clf = HybridClassifier(feedback_path=tmp_path / "fb.csv")
    if clf.model is None:
        pytest.skip("model not trained")
    cat, src, conf, _ = clf.classify("UPI/CR/412345678901/RAMESH IYER/SBIN/rameshiyer@oksbi/NA", "credit")
    assert src == "ml" and cat == "Transfer In"
    cat, src, _, _ = clf.classify("IMPS/P2A/412345678901/SUNITA PILLAI/HDFC", "debit")
    assert cat == "Transfer Out"


def test_feedback_memory(tmp_path):
    clf = HybridClassifier(feedback_path=tmp_path / "fb.csv")
    desc = "UPI/DR/111122223333/RAMESH KIRANA STORES/SBIN/rk@oksbi/NA"
    clf.add_feedback(desc, "debit", "Groceries")
    # same merchant, different reference number -> remembered
    t = Transaction(None, "UPI/DR/999988887777/RAMESH KIRANA STORES/SBIN/rk@oksbi/NA", debit=250.0)
    clf.classify_all([t])
    assert t.category == "Groceries" and t.category_source == "user"
    # persisted to disk -> a new instance also knows it
    assert HybridClassifier(feedback_path=tmp_path / "fb.csv").classify(desc, "debit")[1] == "user"
