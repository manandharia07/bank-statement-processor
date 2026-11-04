"""
Measures extraction accuracy on the sample statements against their ground-truth CSVs.

Run:  python -m scripts.evaluate_extraction [--only name1,name2]
"""
from __future__ import annotations

import argparse
import csv
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bankparser import process_pdf  # noqa: E402
from synthetic.make_statements import PASSWORD  # noqa: E402

SAMPLES = ROOT / "samples"


def load_gt(name):
    with open(SAMPLES / "ground_truth" / f"{name}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def evaluate(name, verbose=False):
    data = (SAMPLES / f"{name}.pdf").read_bytes()
    res = process_pdf(data, name, password=PASSWORD if "protected" in name else None)
    gt = load_gt(name)
    got = res.transactions
    n = min(len(gt), len(got))
    stats = {"date": 0, "debit": 0, "credit": 0, "balance": 0, "desc_exact": 0, "desc_sim": 0.0}
    for g, t in zip(gt, got):
        stats["date"] += (t.date.isoformat() if t.date else "") == g["date"]
        stats["debit"] += abs(t.debit - float(g["debit"])) < 0.005
        stats["credit"] += abs(t.credit - float(g["credit"])) < 0.005
        stats["balance"] += t.balance is not None and abs(t.balance - float(g["balance"])) < 0.005
        stats["desc_exact"] += t.description == g["description"]
        sim = difflib.SequenceMatcher(None, t.description, g["description"]).ratio()
        stats["desc_sim"] += sim
        if verbose and (sim < 0.98 or abs(t.debit - float(g["debit"])) > 0.005):
            print(f"   GT : {g['date']} | {g['description']} | {g['debit']} | {g['credit']} | {g['balance']}")
            print(f"   GOT: {t.date} | {t.description} | {t.debit} | {t.credit} | {t.balance} {t.flags}")
    pct = {k: round(100 * v / n, 1) if n else 0 for k, v in stats.items()}
    print(f"{name:28s} rows {len(got):3d}/{len(gt):3d}  date {pct['date']}%  debit {pct['debit']}%  "
          f"credit {pct['credit']}%  balance {pct['balance']}%  desc-exact {pct['desc_exact']}%  "
          f"desc-sim {pct['desc_sim']}%  reconciled {res.reconciliation.get('match_rate')}%  "
          f"parser={res.parser}  pages={[p.kind for p in res.pages]}")
    if verbose:
        print("   account:", res.account)
        print("   warnings:", res.warnings)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    names = [p.stem for p in sorted(SAMPLES.glob("*.pdf")) if p.stem != "not_a_statement"]
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    for n in names:
        try:
            evaluate(n, a.v)
        except Exception as exc:
            print(f"{n:28s} ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
