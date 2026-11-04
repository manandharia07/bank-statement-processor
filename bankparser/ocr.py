"""
OCR for image-based (scanned) pages using Tesseract.

Steps: render page -> fix 90/180 degree rotation (Tesseract OSD) -> deskew small tilt
-> clean up (contrast + Otsu threshold) -> Tesseract word boxes -> convert to Word objects
in PDF point coordinates (same coordinate system as native text, so the parser is shared).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageOps

from .models import Word

_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


def find_tesseract() -> str | None:
    """Locate the tesseract binary (env var > PATH > default Windows install folders)."""
    for cand in [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"), *_WINDOWS_PATHS]:
        if cand and Path(cand).exists():
            pytesseract.pytesseract.tesseract_cmd = cand
            return cand
    return None


def tesseract_available() -> bool:
    return find_tesseract() is not None


def _otsu(arr: np.ndarray) -> int:
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = sum_b = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0 or w_b == total:
            continue
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / (total - w_b)
        between = w_b * (total - w_b) * (m_b - m_f) ** 2
        if between > best:
            best, thr = between, t
    return thr


def estimate_skew(img: Image.Image) -> float:
    """Projection-profile deskew: the rotation that makes text rows sharpest
    (largest variance between adjacent row sums) is the correction angle."""
    small = img.copy()
    small.thumbnail((900, 900))
    arr = np.asarray(small)
    ink = Image.fromarray(((arr < _otsu(arr)) * 255).astype(np.uint8))

    def score(angle):
        rot = np.asarray(ink.rotate(angle, resample=Image.NEAREST, fillcolor=0), dtype=np.float32)
        rows = rot.sum(axis=1)
        return float(np.sum((rows[1:] - rows[:-1]) ** 2))

    coarse = max(np.arange(-3.0, 3.01, 0.5), key=score)
    fine = max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=score)
    return round(float(fine), 2)


def _ink(img: Image.Image, size: int = 900) -> np.ndarray:
    small = img.copy()
    small.thumbnail((size, size))
    arr = np.asarray(small)
    return (arr < _otsu(arr)).astype(np.float32)


def looks_sideways(img: Image.Image) -> bool:
    """Cheap test (no OCR): text lines make the ROW profile of a page sharply striped.
    If the COLUMN profile is sharper, the text runs vertically -> page is rotated 90/270."""
    ink = _ink(img)
    rows, cols = ink.sum(axis=1), ink.sum(axis=0)
    row_score = np.mean(np.diff(rows) ** 2) / max(ink.shape[1], 1) ** 2
    col_score = np.mean(np.diff(cols) ** 2) / max(ink.shape[0], 1) ** 2
    return col_score > row_score


def _orientation(img: Image.Image) -> int:
    """Tesseract OSD: degrees the page must be rotated (0/90/180/270). Expensive - call only when needed."""
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        if float(osd.get("orientation_conf", 0)) > 2:
            return int(osd.get("rotate", 0))
    except pytesseract.TesseractError:
        pass  # too little text for OSD
    return 0


def _read(img: Image.Image, page: fitz.Page, page_no: int, meta: dict):
    # small tilt
    angle = estimate_skew(img)
    if abs(angle) >= 0.1:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    meta["skew"] = angle if abs(angle) >= 0.1 else 0.0
    # clean up: stretch contrast, then binarise
    img = ImageOps.autocontrast(img, cutoff=1)
    arr = np.asarray(img)
    img = Image.fromarray(np.where(arr > _otsu(arr), 255, 0).astype(np.uint8))
    del arr
    # words with boxes; psm 6 = treat page as one uniform block (works well for tables)
    data = pytesseract.image_to_data(img, config="--oem 1 --psm 6 -c preserve_interword_spaces=1",
                                     output_type=pytesseract.Output.DICT)
    # image pixels -> PDF points (a 90-degree rotation swaps the canvas, so scale per axis)
    sx = page.rect.width / img.width
    sy = page.rect.height / img.height
    words, confs = [], []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        if conf < 15 and not any(ch.isalnum() for ch in text):
            continue  # specks of noise
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(Word(text, x * sx, (x + w) * sx, y * sy, (y + h) * sy, page_no, conf))
        confs.append(conf)
    meta["confidence"] = round(float(np.mean(confs)), 1) if confs else 0.0
    return words


def ocr_page(page: fitz.Page, page_no: int, dpi: int = 300):
    """Returns (words, meta) for one page.

    Orientation detection (Tesseract OSD) costs as much as a second OCR pass, so it only runs when
    the page looks sideways (cheap projection test) or the first OCR pass has very low confidence
    (typical for an upside-down scan)."""
    if not find_tesseract():
        raise RuntimeError("Tesseract OCR is not installed - it is required for scanned pages. "
                           "Install it (see README) or set TESSERACT_CMD.")
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    del pix
    meta = {"rotation": 0, "skew": 0.0}

    checked = False
    if looks_sideways(img):
        rot, checked = _orientation(img), True
        if rot:
            img = img.rotate(-rot, expand=True, fillcolor=255)
            meta["rotation"] = rot
    words = _read(img, page, page_no, meta)
    if not checked and meta["confidence"] < LOW_CONFIDENCE_RETRY:
        rot = _orientation(img)
        if rot:
            meta["rotation"] = rot
            words = _read(img.rotate(-rot, expand=True, fillcolor=255), page, page_no, meta)
    return words, meta


LOW_CONFIDENCE_RETRY = 55
