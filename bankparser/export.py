"""
Export to Excel (.xlsx, multi-sheet, formatted) and CSV.

Excel sheets:
  Summary       account details + reconciliation results + processing info
  Transactions  every row with category, confidence, source, validation flags (problem rows highlighted)
  Categories    total debit / credit / count per category
  Monthly       money in / out per month
"""
from __future__ import annotations

import io

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ACCOUNT_LABELS = {
    "bank_name": "Bank", "account_holder": "Account Holder", "account_number": "Account Number",
    "ifsc": "IFSC", "micr": "MICR", "customer_id": "Customer ID", "branch": "Branch",
    "account_type": "Account Type", "currency": "Currency", "period_from": "Statement From",
    "period_to": "Statement To", "opening_balance": "Opening Balance (stated)",
    "closing_balance": "Closing Balance (stated)",
}


def transactions_df(results) -> pd.DataFrame:
    """One DataFrame for one or many StatementResults."""
    rows = []
    for res in results:
        acct = res.account.get("account_number", "")
        for t in res.transactions:
            rows.append({
                "Date": pd.Timestamp(t.date) if t.date else pd.NaT,
                "Value Date": pd.Timestamp(t.value_date) if t.value_date else pd.NaT,
                "Description": t.description,
                "Ref / Cheque No": t.ref,
                "Debit": round(t.debit, 2) if t.debit else None,
                "Credit": round(t.credit, 2) if t.credit else None,
                "Balance": t.balance,
                "Category": t.category,
                "Category Source": t.category_source,
                "Confidence": t.confidence,
                "Category Reason": t.category_reason,
                "Validation Flags": "; ".join(t.flags),
                "Page": t.page,
                "Account Number": acct,
                "Source File": res.filename,
            })
    return pd.DataFrame(rows)


def to_csv(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in ("Date", "Value Date"):
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8-sig")       # BOM so Excel opens ₹/unicode correctly


def _summary_rows(results):
    rows = []
    for res in results:
        rows.append(("File", res.filename))
        for k, label in ACCOUNT_LABELS.items():
            if k in res.account:
                rows.append((label, res.account[k]))
        r = res.reconciliation
        rows += [
            ("Transactions", len(res.transactions)),
            ("Total Debit", r.get("total_debit")),
            ("Total Credit", r.get("total_credit")),
            ("Opening Balance (used)", r.get("opening_balance")),
            ("Closing Balance (computed)", r.get("closing_balance")),
            ("Balance Reconciliation", f"{r.get('rows_matched')}/{r.get('rows_checked')} rows "
                                       f"({r.get('match_rate')}%)"),
            ("Parser", res.parser),
            ("Row Order", res.order),
            ("Pages", ", ".join(f"p{p.number}:{p.kind}" + (f"({p.ocr_confidence}%)" if p.kind == "ocr" else "")
                                for p in res.pages)),
            ("Warnings", " | ".join(res.warnings) or "none"),
            ("", ""),
        ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def to_excel(results, df: pd.DataFrame | None = None) -> bytes:
    df = transactions_df(results) if df is None else df
    cat = (df.groupby("Category", dropna=False)
             .agg(Transactions=("Description", "count"), Total_Debit=("Debit", "sum"), Total_Credit=("Credit", "sum"))
             .reset_index().sort_values("Total_Debit", ascending=False))
    cat.columns = ["Category", "Transactions", "Total Debit", "Total Credit"]
    month = df.dropna(subset=["Date"]).assign(Month=lambda d: d["Date"].dt.to_period("M").astype(str))
    month = (month.groupby("Month").agg(Money_In=("Credit", "sum"), Money_Out=("Debit", "sum"),
                                        Transactions=("Description", "count")).reset_index())
    month["Net"] = month["Money_In"] - month["Money_Out"]
    month.columns = ["Month", "Money In", "Money Out", "Transactions", "Net"]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl", date_format="DD-MM-YYYY", datetime_format="DD-MM-YYYY") as xw:
        _summary_rows(results).to_excel(xw, sheet_name="Summary", index=False)
        df.to_excel(xw, sheet_name="Transactions", index=False)
        cat.to_excel(xw, sheet_name="Categories", index=False)
        month.to_excel(xw, sheet_name="Monthly", index=False)
        _style(xw, df)
    return buf.getvalue()


def _style(xw, df):
    head_fill = PatternFill("solid", fgColor="1F4E78")
    warn_fill = PatternFill("solid", fgColor="FCE4D6")
    review_fill = PatternFill("solid", fgColor="FFF2CC")
    money = '#,##,##0.00'                      # Indian grouping
    for name, ws in xw.sheets.items():
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = head_fill
            cell.alignment = Alignment(vertical="center")
        ws.freeze_panes = "A2"
        for i, col in enumerate(ws.columns, 1):
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 60)
        if name in ("Categories", "Monthly", "Transactions"):
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, float):
                        c.number_format = money
        if name == "Transactions" and len(df):
            ws.auto_filter.ref = ws.dimensions
            cols = {c.value: c.column for c in ws[1]}
            for r_idx, (_, r) in enumerate(df.iterrows(), start=2):
                fill = None
                if "mismatch" in str(r["Validation Flags"]) or "unknown" in str(r["Validation Flags"]):
                    fill = warn_fill
                elif r["Category"] == "Uncategorized":
                    fill = review_fill
                if fill:
                    for c in range(1, ws.max_column + 1):
                        ws.cell(r_idx, c).fill = fill
                ws.cell(r_idx, cols["Confidence"]).number_format = "0%"
