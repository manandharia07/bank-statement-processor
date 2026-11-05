"""Bank Statement Processing & Classification System - Streamlit UI: upload bank statements, review extraction + classification, correct, export.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from bankparser import PasswordRequired, StatementError, WrongPassword, process_pdf
from bankparser.classify.hybrid import HybridClassifier
from bankparser.export import ACCOUNT_LABELS, to_csv, to_excel, transactions_df
from bankparser.normalize import fmt_inr
from bankparser.ocr import tesseract_available

st.set_page_config(page_title="Bank Statement Processing & Classification System", page_icon="🏦", layout="wide")
SAMPLES = Path(__file__).parent / "samples"
SAMPLE_PASSWORD = "RAHU1990"     # password of the generated sample samples/sahyadri_bank_protected.pdf


@st.cache_resource
def get_classifier() -> HybridClassifier:
    return HybridClassifier()


def file_key(f) -> str:
    return hashlib.md5(f.getvalue()).hexdigest()[:10] + "_" + f.name


ss = st.session_state
ss.setdefault("results", {})        # file_key -> StatementResult
ss.setdefault("errors", {})         # file_key -> message
ss.setdefault("need_pw", {})        # file_key -> True

clf = get_classifier()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    force_ocr = st.toggle("Force OCR on every page", value=False,
                          help="Use when a PDF has a broken text layer.")
    dpi = st.select_slider("OCR resolution (DPI)", [200, 250, 300, 400], value=300)
    clf.threshold = st.slider("ML confidence threshold", 0.1, 0.9, 0.40, 0.05,
                              help="Below this the row is marked Uncategorized for review.")
    st.divider()
    st.caption("System status")
    st.write("✅ Tesseract OCR found" if tesseract_available() else "⚠️ Tesseract not found - scanned PDFs disabled")
    st.write("✅ ML model loaded" if clf.model else "⚠️ ML model missing - run training")
    st.write(f"🧠 Learned corrections: {len(clf.memory)}")
    if st.button("Retrain ML model with my corrections", use_container_width=True):
        with st.spinner("Retraining..."):
            info = clf.retrain()
            for res in ss.results.values():
                clf.classify_all(res.transactions)
        st.success(f"Retrained on {info['rows']} synthetic + {info['feedback_rows']} corrected rows.")
    if st.button("Clear all results", use_container_width=True):
        ss.results, ss.errors, ss.need_pw = {}, {}, {}
        st.rerun()

st.title("🏦 Bank Statement Processing & Classification System")
st.caption("Text & scanned PDFs · any bank layout · balance-verified extraction · "
           "rule + ML classification (no LLM) · Excel / CSV export")

# ------------------------------------------------------------------ upload
def run_one(k: str, name: str, data: bytes, password: str | None, bar, i: int, n: int):
    def progress(p, total, msg):
        bar.progress(min(1.0, (i + p / max(total, 1)) / n), text=f"{name}: {msg}")

    t0 = time.time()
    try:
        res = process_pdf(data, name, password=password or None, force_ocr=force_ocr, dpi=dpi,
                          classifier=clf, progress=progress)
        res.seconds = round(time.time() - t0, 1)
        ss.results[k] = res
        ss.errors.pop(k, None)
        ss.need_pw.pop(k, None)
    except (PasswordRequired, WrongPassword) as e:
        ss.need_pw[k] = True
        ss.errors[k] = f"{name}: {e}"
    except StatementError as e:
        ss.errors[k] = f"{name}: {e}"
    except Exception as e:  # never crash the UI
        ss.errors[k] = f"{name}: unexpected error - {type(e).__name__}: {e}"


up_col, sample_col = st.columns([3, 2])
with up_col:
    files = st.file_uploader("Upload one or more bank statement PDFs", type=["pdf"], accept_multiple_files=True)
    passwords = {}
    if files:
        for f in files:
            k = file_key(f)
            if ss.need_pw.get(k):
                passwords[k] = st.text_input(f"🔒 Password for **{f.name}**", type="password", key=f"pw_{k}")
        if st.button("Process statements", type="primary"):
            bar = st.progress(0.0, text="Starting...")
            for i, f in enumerate(files):
                k = file_key(f)
                if k not in ss.results or force_ocr:
                    run_one(k, f.name, f.getvalue(), passwords.get(k), bar, i, len(files))
            bar.empty()
            st.rerun()
with sample_col:
    with st.container(border=True):
        st.markdown("**No statement handy? Try a sample**")
        names = sorted(p.name for p in SAMPLES.glob("*.pdf"))
        pick = st.selectbox("Sample statement", names, label_visibility="collapsed")
        if st.button("Process sample"):
            bar = st.progress(0.0, text="Starting...")
            run_one("sample_" + pick, pick, (SAMPLES / pick).read_bytes(),
                    SAMPLE_PASSWORD if "protected" in pick else None, bar, 0, 1)
            bar.empty()
            st.rerun()
        st.caption("Fictional banks generated by `synthetic/make_statements.py`: text, scanned, "
                   "mixed, password-protected and a non-statement PDF.")

for msg in ss.errors.values():
    st.error(msg)

results = list(ss.results.values())
if not results:
    st.info("Upload a PDF and press **Process statements**, or process one of the samples on the right.")
    st.stop()

# ------------------------------------------------------------------ overview
tab_over, tab_txn, tab_ins, tab_exp = st.tabs(["📋 Overview", "📑 Transactions", "📊 Insights", "⬇️ Export"])

with tab_over:
    for res in results:
        r = res.reconciliation
        with st.container(border=True):
            st.subheader(f"{res.account.get('bank_name', 'Statement')} — {res.filename}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Transactions", len(res.transactions))
            c2.metric("Total Debit", f"₹{fmt_inr(r.get('total_debit'))}")
            c3.metric("Total Credit", f"₹{fmt_inr(r.get('total_credit'))}")
            c4.metric("Balance check", f"{r.get('match_rate')}%" if r.get("match_rate") is not None else "n/a",
                      help="Rows where previous balance + credit - debit = printed balance")
            c5.metric("Processed in", f"{getattr(res, 'seconds', '?')} s")
            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Account details**")
                acct = [(ACCOUNT_LABELS.get(k, k), fmt_inr(v) if isinstance(v, float) else v)
                        for k, v in res.account.items()]
                st.table(pd.DataFrame(acct, columns=["Field", "Value"]).set_index("Field"))
            with right:
                st.markdown("**Page detection (auto)**")
                st.dataframe(pd.DataFrame([{
                    "Page": p.number, "Type": "📄 text" if p.kind == "text" else "🖼️ scanned → OCR",
                    "Why": p.reason, "OCR conf.": p.ocr_confidence, "Deskew°": p.skew or None,
                    "Rows": p.transactions} for p in res.pages]), hide_index=True, use_container_width=True)
                st.markdown(f"**Parser:** {res.parser} · **Order:** {res.order}")
                st.markdown(f"**Opening → Closing:** ₹{fmt_inr(r.get('opening_balance'))} → "
                            f"₹{fmt_inr(r.get('closing_balance'))}")
            for w in res.warnings:
                st.warning(w)

# ------------------------------------------------------------------ transactions (editable)
index = [(k, i) for k, res in ss.results.items() for i in range(len(res.transactions))]
df = transactions_df(results)
if len(results) > 1:
    before = len(df)
    dup_mask = df.duplicated(subset=["Account Number", "Date", "Description", "Debit", "Credit", "Balance"])
    df, index = df[~dup_mask].reset_index(drop=True), [x for x, d in zip(index, dup_mask) if not d]
    if before - len(df):
        st.toast(f"{before - len(df)} overlapping transaction(s) across files removed")

with tab_txn:
    f1, f2, f3 = st.columns([1, 1, 2])
    only_review = f1.toggle("Only rows needing review")
    cat_filter = f2.multiselect("Category", sorted(df["Category"].unique()))
    search = f3.text_input("Search description")
    view = df.copy()
    view["Needs Review"] = (view["Category"].eq("Uncategorized") |
                            view["Validation Flags"].str.contains("mismatch|unknown|inferred", na=False))
    mask = pd.Series(True, index=view.index)
    if only_review:
        mask &= view["Needs Review"]
    if cat_filter:
        mask &= view["Category"].isin(cat_filter)
    if search:
        mask &= view["Description"].str.contains(search, case=False, na=False, regex=False)
    shown = view[mask]
    st.caption(f"{len(shown)} of {len(view)} rows · edit the **Category** column to correct it, then save. "
               "Corrections are remembered and can be used to retrain the model.")
    edited = st.data_editor(
        shown[["Date", "Description", "Debit", "Credit", "Balance", "Category", "Category Source",
               "Confidence", "Category Reason", "Validation Flags", "Needs Review"]],
        column_config={
            "Date": st.column_config.DateColumn(format="DD-MM-YYYY"),
            "Debit": st.column_config.NumberColumn(format="%.2f"),
            "Credit": st.column_config.NumberColumn(format="%.2f"),
            "Balance": st.column_config.NumberColumn(format="%.2f"),
            "Category": st.column_config.SelectboxColumn(options=clf.categories, required=True),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        },
        disabled=[c for c in shown.columns if c != "Category"],
        hide_index=True, use_container_width=True, height=520, key="editor")
    changed = edited.index[edited["Category"] != shown["Category"]]
    if len(changed):
        if st.button(f"💾 Save {len(changed)} correction(s)", type="primary"):
            for ix in changed:
                k, ti = index[ix]
                t = ss.results[k].transactions[ti]
                direction = "credit" if t.credit > 0 and not t.debit else "debit"
                clf.add_feedback(t.description, direction, edited.at[ix, "Category"])
            for res in ss.results.values():        # apply learned corrections to similar rows too
                clf.classify_all(res.transactions)
            st.success("Saved. Similar transactions were updated as well.")
            st.rerun()

# ------------------------------------------------------------------ insights
with tab_ins:
    spend = df[df["Debit"].notna()].groupby("Category")["Debit"].sum().sort_values(ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Spending by category (₹)**")
        st.bar_chart(spend, horizontal=True, color="#1F4E78")
    with c2:
        st.markdown("**Money in vs out by month (₹)**")
        m = df.dropna(subset=["Date"]).assign(Month=lambda d: d["Date"].dt.strftime("%Y-%m"))
        monthly = m.groupby("Month")[["Credit", "Debit"]].sum().rename(columns={"Credit": "In", "Debit": "Out"})
        st.bar_chart(monthly, stack=False, color=["#2E8B57", "#C0504D"])
    src = df["Category Source"].value_counts()
    st.markdown("**How rows were classified:** " + " · ".join(f"{k}: {v}" for k, v in src.items()))
    st.markdown("**Top 10 debits**")
    st.dataframe(df.nlargest(10, "Debit")[["Date", "Description", "Debit", "Category"]], hide_index=True,
                 use_container_width=True)

# ------------------------------------------------------------------ export
with tab_exp:
    base = results[0].filename.rsplit(".", 1)[0] if len(results) == 1 else "statements"
    st.download_button("⬇️ Download Excel (.xlsx)", to_excel(results, df), f"{base}_processed.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.download_button("⬇️ Download CSV (.csv)", to_csv(df), f"{base}_processed.csv", "text/csv")
    st.caption("Excel contains 4 sheets: Summary (account + reconciliation), Transactions (highlighted rows need "
               "review), Categories, Monthly. CSV contains the transactions table.")
