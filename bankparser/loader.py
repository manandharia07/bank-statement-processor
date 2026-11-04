"""
Opening PDFs and turning every page into a list of Words.

Auto-detection is done PER PAGE, because real statements can mix native-text pages and
scanned pages in a single file.
"""
from __future__ import annotations

import fitz

from .models import PageInfo, Word
from .ocr import ocr_page


class StatementError(Exception):
    """Base class for user-facing errors (shown as friendly messages in the UI)."""


class PasswordRequired(StatementError):
    pass


class WrongPassword(StatementError):
    pass


class InvalidPDF(StatementError):
    pass


class NotAStatement(StatementError):
    pass


MAX_PAGES = 200


def open_pdf(data: bytes, password: str | None = None) -> fitz.Document:
    if not data or not data[:1024].lstrip().startswith(b"%PDF"):
        raise InvalidPDF("The file is not a PDF (missing %PDF header).")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # corrupted / truncated
        raise InvalidPDF(f"The PDF could not be opened - it may be corrupted ({exc}).") from exc
    if doc.needs_pass:
        if not password:
            raise PasswordRequired("This PDF is password protected. Please enter the password "
                                   "(banks often use DOB / PAN / customer-ID based passwords).")
        if not doc.authenticate(password):
            raise WrongPassword("Incorrect password for this PDF.")
    if doc.page_count == 0:
        raise InvalidPDF("The PDF has no pages.")
    if doc.page_count > MAX_PAGES:
        raise InvalidPDF(f"The PDF has {doc.page_count} pages; the limit is {MAX_PAGES}.")
    return doc


def detect_page_kind(page: fitz.Page) -> tuple[str, str]:
    """Decide whether a page needs OCR. Returns (kind, reason)."""
    raw = page.get_text("words")
    text = "".join(w[4] for w in raw)
    n = len(text)
    if n < 40:
        return "ocr", f"only {n} text characters - page is an image"
    readable = sum(ch.isalnum() for ch in text) / n
    if readable < 0.5:
        return "ocr", f"text layer looks corrupted ({readable:.0%} readable)"
    page_area = abs(page.rect)
    img_area = 0.0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        img_area += abs(r)
    if page_area and img_area / page_area > 0.6 and len(raw) < 60:
        return "ocr", "large scanned image with little text"
    return "text", f"{len(raw)} words in text layer"


def text_words(page: fitz.Page, page_no: int) -> list[Word]:
    return [Word(w[4], w[0], w[2], w[1], w[3], page_no) for w in page.get_text("words") if w[4].strip()]


def extract_pages(doc: fitz.Document, force_ocr: bool = False, dpi: int = 300, progress=None):
    """Yields (PageInfo, words) for every page."""
    for i, page in enumerate(doc):
        no = i + 1
        kind, reason = ("ocr", "OCR forced by user") if force_ocr else detect_page_kind(page)
        if progress:
            progress(i, doc.page_count, f"Page {no}: {'running OCR' if kind == 'ocr' else 'reading text'}")
        if kind == "text":
            words = text_words(page, no)
            info = PageInfo(no, "text", reason, len(words))
        else:
            words, meta = ocr_page(page, no, dpi=dpi)
            info = PageInfo(no, "ocr", reason, len(words), meta.get("confidence"),
                            meta.get("rotation", 0), meta.get("skew", 0.0))
        yield info, words
