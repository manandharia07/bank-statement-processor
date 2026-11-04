"""
Command-line interface (batch / automation use, no UI needed).

  python -m bankparser statement.pdf [more.pdf ...] -o result.xlsx [--csv result.csv] [--password X] [--force-ocr]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import StatementError, process_pdf
from .classify.hybrid import HybridClassifier
from .export import to_csv, to_excel, transactions_df


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bankparser", description="Extract + classify bank statement PDFs.")
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("statement_processed.xlsx"))
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--password")
    ap.add_argument("--force-ocr", action="store_true")
    a = ap.parse_args(argv)

    clf = HybridClassifier()
    results = []
    for p in a.pdfs:
        try:
            res = process_pdf(p.read_bytes(), p.name, password=a.password, force_ocr=a.force_ocr, classifier=clf)
        except StatementError as e:
            print(f"[skip] {p.name}: {e}", file=sys.stderr)
            continue
        r = res.reconciliation
        print(f"[ok]   {p.name}: {len(res.transactions)} transactions, pages={[pg.kind for pg in res.pages]}, "
              f"balance check {r['rows_matched']}/{r['rows_checked']}")
        for w in res.warnings:
            print(f"       ! {w}")
        results.append(res)
    if not results:
        return 1
    df = transactions_df(results)
    a.output.write_bytes(to_excel(results, df))
    print(f"Excel -> {a.output}")
    if a.csv:
        a.csv.write_bytes(to_csv(df))
        print(f"CSV   -> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
