"""
PDF Outliner (standalone)
=========================
Adds bookmark outlines to a scanned exam-paper PDF.

It opens the PDF, OCRs every page, and builds:

  * Top-level outline entries per academic year
    (``Final Examination 2022-2023``, ``Final Examination 2021-2022`` …)
  * Sub-entries ``Question 1`` … ``Question 8`` under each year, each
    linking to the page where that question begins.

Two OCR backends are supported (pick with ``--backend``):

  * ``tesseract`` *(default)* — local OCR via pytesseract.  Free, runs
    on CPU, no network needed.  Question / sub-part / year detection is
    threshold-based; cycle inference recovers questions whose marker
    OCR missed but whose ``(a)/(b)/(c)`` sub-parts are still visible.
  * ``chandra`` — Datalab's hosted Chandra OCR
    (https://www.datalab.to).  State-of-the-art OCR for academic
    layouts, ~$0.005-0.01 per page.  Requires an API key in the
    ``DATALAB_API_KEY`` env var (or via ``--api-key``).  Submits the
    PDF, polls until done, parses the paginated Markdown response.

Edit the CONFIG block below, then run:    python add_outlines.py
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from pathlib import Path

import fitz                  # PyMuPDF

# Use UTF-8 for console output (Windows console defaults to cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  —  edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Input PDF (the scanned question paper).
PDF_PATH = r"C:\Users\Adit Al Razi\Downloads\babas\EEE 317.pdf"

# Where to write the bookmarked PDF.  Must be a different path than PDF_PATH.
OUTPUT_PDF = r"C:\Users\Adit Al Razi\Downloads\babas\EEE 317 - outlined.pdf"

# Which OCR backend to use by default ("tesseract" or "chandra").  Can be
# overridden on the command line with --backend.
DEFAULT_BACKEND = "tesseract"

# ── Tesseract backend ──────────────────────────────────────────────────────
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DPI = 220
SKIP_PAGES: list[int] = []

# ── Chandra (Datalab API) backend ──────────────────────────────────────────
DATALAB_API_BASE = "https://www.datalab.to"
DATALAB_POLL_INTERVAL_S = 3
DATALAB_MAX_WAIT_S = 600        # 10 min hard cap

# ── Outline shape (shared) ─────────────────────────────────────────────────
MAX_QUESTION_FOR_OUTLINE = 8

# Title templates.  Placeholders: {year}, {num}.
YEAR_TITLE_TEMPLATE     = "{year}"
QUESTION_TITLE_TEMPLATE = "Question {num}"

# ── Tesseract detection thresholds (fractions of each page's pixel size) ──
LEFT_MARGIN_MIN_FRAC   = 0.05
LEFT_MARGIN_MAX_FRAC   = 0.14
MIN_MARKER_HEIGHT_FRAC = 0.0055
MAX_MARKER_HEIGHT_FRAC = 0.025
MIN_MARKER_WIDTH_FRAC  = 0.005
MIN_CONFIDENCE         = 20
LOOSE_MIN_CONFIDENCE   = 60
TOP_FRACTION           = 0.045
BOTTOM_FRACTION        = 0.955
MAX_QUESTION_NUMBER    = 15

SUBPART_MIN_FRAC       = 0.11
SUBPART_MAX_FRAC       = 0.25
SUBPART_MIN_WIDTH_FRAC = 0.008
SUBPART_MIN_CONF       = 30
SUBPART_LETTERS        = "abcdefgh"
MIN_SUBPART_GAP_FRAC   = 0.020
SAME_LINE_TOL_FRAC     = 0.013
MAX_QUESTION_PAGES     = 2

# Year-header detection (shared between backends)
YEAR_RE = re.compile(r"(20\d{2})\s*[-–—/]\s*(20\d{2}|\d{2})")
YEAR_SEARCH_TOP_FRAC = 0.15                    # tesseract only
YEAR_TEXT_HEAD_CHARS = 1500                    # chandra: search top N chars

# ─────────────────────────────────────────────────────────────────────────────


# Question-marker patterns (tesseract path)
RE_STRICT  = re.compile(r"^(\d{1,2})\s*[\.\,\:\;]$")
RE_LOOSE   = re.compile(r"^(\d{1,2})$")
RE_ONE_FIX = re.compile(r"^[il|I!]\s*[\.\,\:\;]?$")
RE_COMBO   = re.compile(r"^(\d{1,2})\s*[\.\,\:\;]?\s*\(\s*([a-z])\s*\)\s*$")
ROMAN_SKIP = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}

# Question marker at the start of a markdown line (chandra path).
# Allows leading whitespace, blockquote/bullet markers; requires "N." or "N)".
QUESTION_TEXT_RE = re.compile(r"^[\s>*\-]*(\d{1,2})\s*[\.\)]\s", re.MULTILINE)

# Marker's "paginate: true" delimiter between pages.  Marker emits
# ``\n\n{n}-{48 dashes}\n\n`` between pages by default; the braces and
# exact dash count vary slightly between versions, so we're lenient.
PAGE_DELIM_RE = re.compile(r"\n+\{?(\d+)\}?[-=_]{8,}\n+")


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers (used by both backends)
# ═════════════════════════════════════════════════════════════════════════════


def parse_year_string(text: str, head_chars: int = YEAR_TEXT_HEAD_CHARS) -> str | None:
    """Find an academic-year tag near the top of a text blob.

    Accepts ``2022-2023`` and ``2022-23`` forms.  Rejects spans that are
    not 0 or 1 calendar year wide (filters OCR noise like ``2013-24``).
    """
    head = text[:head_chars]
    for m in YEAR_RE.finditer(head):
        y1 = int(m.group(1))
        raw_y2 = m.group(2)
        if len(raw_y2) == 2:
            y2_full = (y1 // 100) * 100 + int(raw_y2)
            if y2_full < y1:
                y2_full += 100
        else:
            y2_full = int(raw_y2)
        if y2_full - y1 not in (0, 1):
            continue
        return f"{y1}-{y2_full % 100:02d}"
    return None


def find_questions_in_markdown(text: str, max_q: int) -> list[int]:
    """Return sorted, unique question numbers (1..max_q) appearing as
    list markers at the start of any line in ``text``."""
    nums: set[int] = set()
    for m in QUESTION_TEXT_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= max_q:
            nums.add(n)
    return sorted(nums)


def year_label(year_tag: str) -> str:
    """Expand "2022-23" → "2022-2023".  Unexpected formats pass through."""
    parts = year_tag.split("-")
    if len(parts) != 2:
        return year_tag
    y1_str, y2_str = parts
    if not (y1_str.isdigit() and y2_str.isdigit()):
        return year_tag
    y1 = int(y1_str)
    if len(y2_str) == 2:
        century = (y1 // 100) * 100
        y2_full = century + int(y2_str)
        if y2_full < y1:
            y2_full += 100
        return f"{y1}-{y2_full}"
    return year_tag


def _drop_out_of_order(q_pages: dict[int, int]) -> dict[int, int]:
    """Drop entries where a lower-numbered question's page is *after* a
    higher-numbered one's — almost always an OCR false positive (an
    "i" or "l" in body text read as "1.")."""
    if len(q_pages) < 2:
        return q_pages
    valid = dict(q_pages)
    min_after = float("inf")
    for q_num in sorted(valid.keys(), reverse=True):
        page = valid[q_num]
        if page > min_after:
            del valid[q_num]
        else:
            min_after = page
    return valid


def build_toc(
    per_page: list[tuple[int, str | None, list[int]]],
    max_q: int,
) -> list[list]:
    """Build a PyMuPDF TOC ([level, title, page_1idx]) from per-page
    detections.  Each entry in ``per_page`` is
    ``(page_idx_0idx, year_detected_on_this_page_or_None, [q_nums])``.

    The most recently seen year carries forward to pages that don't
    have their own header.  Within a year, questions whose detected
    page would go backwards (likely OCR noise) are dropped.
    """
    years_in_order: list[str] = []
    year_first_page: dict[str, int] = {}
    by_year: dict[str, dict[int, int]] = {}
    current_year: str | None = None

    for page_idx, year, q_nums in per_page:
        if year:
            current_year = year
            if year not in year_first_page:
                year_first_page[year] = page_idx
                years_in_order.append(year)
                by_year[year] = {}
        if not current_year:
            continue
        q_pages = by_year.setdefault(current_year, {})
        for n in q_nums:
            if n > max_q:
                continue
            if n not in q_pages or page_idx < q_pages[n]:
                q_pages[n] = page_idx

    toc: list[list] = []
    for year in years_in_order:
        q_pages = _drop_out_of_order(by_year.get(year, {}))
        toc.append([
            1,
            YEAR_TITLE_TEMPLATE.format(year=year_label(year)),
            year_first_page[year] + 1,
        ])
        for q_num in sorted(q_pages.keys()):
            toc.append([
                2,
                QUESTION_TITLE_TEMPLATE.format(num=q_num),
                q_pages[q_num] + 1,
            ])
    return toc


# ═════════════════════════════════════════════════════════════════════════════
# Tesseract backend
# ═════════════════════════════════════════════════════════════════════════════


def parse_marker(text: str) -> tuple[int, str] | None:
    t = text.strip()
    if not t:
        return None
    if (m := RE_STRICT.match(t)):
        n = int(m.group(1))
        if 1 <= n <= MAX_QUESTION_NUMBER:
            return n, "strict"
    if (m := RE_LOOSE.match(t)):
        n = int(m.group(1))
        if 1 <= n <= MAX_QUESTION_NUMBER:
            return n, "loose"
    if RE_ONE_FIX.match(t):
        return 1, "fix1"
    return None


def parse_combo(text: str) -> tuple[int, str] | None:
    t = text.strip()
    if not t:
        return None
    m = RE_COMBO.match(t)
    if not m:
        return None
    n = int(m.group(1))
    letter = m.group(2).lower()
    if not (1 <= n <= MAX_QUESTION_NUMBER) or letter not in SUBPART_LETTERS:
        return None
    return n, letter


def parse_subpart(text: str) -> str | None:
    t = text.strip()
    if not t or ("(" not in t and ")" not in t):
        return None
    inner = t.strip("()").strip()
    if not inner or inner.lower() in ROMAN_SKIP or len(inner) != 1:
        return None
    c = inner.lower()
    return c if c in SUBPART_LETTERS else None


def configure_tesseract() -> None:
    import pytesseract
    if TESSERACT_CMD:
        p = Path(TESSERACT_CMD)
        if p.exists():
            pytesseract.pytesseract.tesseract_cmd = str(p)


def render_page(page, dpi: int):
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def detect_year_tesseract(data: dict, page_height: int) -> str | None:
    """Look in the top YEAR_SEARCH_TOP_FRAC of the page for a year string."""
    top_zone = int(page_height * YEAR_SEARCH_TOP_FRAC)
    parts: list[tuple[int, int, str]] = []
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        if not t or data["top"][i] > top_zone:
            continue
        try:
            conf = int(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf >= 0 and conf < 25:
            continue
        parts.append((data["top"][i], data["left"][i], t))
    parts.sort()
    joined = " ".join(t for _, _, t in parts)
    return parse_year_string(joined, head_chars=len(joined) + 1)


def scan_page(img):
    """OCR a page; return question markers, sub-part markers, and any
    year string found in the header zone."""
    import pytesseract
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    w_px, h_px = img.width, img.height

    lm_min  = int(w_px * LEFT_MARGIN_MIN_FRAC)
    lm_max  = int(w_px * LEFT_MARGIN_MAX_FRAC)
    sp_min  = int(w_px * SUBPART_MIN_FRAC)
    sp_max  = int(w_px * SUBPART_MAX_FRAC)
    min_h   = max(8, int(h_px * MIN_MARKER_HEIGHT_FRAC))
    max_h   = int(h_px * MAX_MARKER_HEIGHT_FRAC)
    min_w   = max(4, int(w_px * MIN_MARKER_WIDTH_FRAC))
    sp_w    = max(6, int(w_px * SUBPART_MIN_WIDTH_FRAC))
    gap     = int(h_px * MIN_SUBPART_GAP_FRAC)
    top_limit = int(h_px * TOP_FRACTION)
    bot_limit = int(h_px * BOTTOM_FRACTION)

    q_out: list[tuple[int, int, str]] = []
    sp_out: list[tuple[int, str]] = []

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        try:
            conf = int(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1

        if y < top_limit or y > bot_limit:
            continue

        if (lm_min <= x <= lm_max and min_h <= h <= max_h
                and not (0 <= conf < MIN_CONFIDENCE)):
            combo = parse_combo(text)
            if combo is not None:
                num, letter = combo
                q_out.append((num, y, "combo"))
                sp_out.append((y, letter))
                continue

        if (lm_min <= x <= lm_max and min_h <= h <= max_h
                and w >= min_w and not (0 <= conf < MIN_CONFIDENCE)):
            parsed = parse_marker(text)
            if parsed is not None:
                num, tier = parsed
                if tier == "loose" and 0 <= conf < LOOSE_MIN_CONFIDENCE:
                    pass
                else:
                    q_out.append((num, y, tier))
                    continue

        if (sp_min <= x <= sp_max and min_h <= h <= max_h
                and w >= sp_w and not (0 <= conf < SUBPART_MIN_CONF)):
            letter = parse_subpart(text)
            if letter is not None:
                sp_out.append((y, letter))

    q_out.sort(key=lambda t: t[1])
    sp_out.sort(key=lambda t: t[0])

    tier_rank = {"strict": 0, "combo": 0, "loose": 1, "fix1": 2}
    cleaned_q: list[tuple[int, int, str]] = []
    for m in q_out:
        if cleaned_q and abs(m[1] - cleaned_q[-1][1]) < 8:
            if tier_rank[m[2]] < tier_rank[cleaned_q[-1][2]]:
                cleaned_q[-1] = m
        else:
            cleaned_q.append(m)

    cleaned_sp: list[tuple[int, str]] = []
    for sp in sp_out:
        if cleaned_sp and sp[0] - cleaned_sp[-1][0] < gap:
            continue
        cleaned_sp.append(sp)

    year = detect_year_tesseract(data, h_px)
    return cleaned_q, cleaned_sp, year


def collect_markers(doc):
    events: list[tuple[int, int, int, str]] = []
    subparts: list[tuple[int, int, str]] = []
    page_heights: list[int] = []
    page_year_raw: list[str | None] = []   # year DETECTED on the page only

    for page_idx in range(len(doc)):
        if (page_idx + 1) in SKIP_PAGES:
            page_heights.append(0)
            page_year_raw.append(None)
            print(f"  page {page_idx + 1}: skipped")
            continue
        img = render_page(doc[page_idx], DPI)
        page_heights.append(img.height)
        q_markers, sp_markers, year = scan_page(img)
        page_year_raw.append(year)

        for num, y, tier in q_markers:
            events.append((page_idx, y, num, tier))
        for y, letter in sp_markers:
            subparts.append((page_idx, y, letter))
        if q_markers or sp_markers or year:
            qs  = ", ".join(f"{n}({t[0]})" for n, _, t in q_markers) or "-"
            sps = "".join(letter for _, letter in sp_markers) or "-"
            year_tag = f"  [year {year}]" if year else ""
            print(f"  page {page_idx + 1}:  Q[{qs}]  sub[{sps}]{year_tag}")
    return events, subparts, page_heights, page_year_raw


def determine_question_end(events, idx: int, n_pages: int):
    s_page, _s_y, s_num, _s_tier = events[idx]
    cap_page = min(s_page + MAX_QUESTION_PAGES - 1, n_pages - 1)
    if idx + 1 >= len(events):
        return (s_page, None)
    n_page, n_y, n_num, _n_tier = events[idx + 1]
    if n_num <= s_num:
        return (s_page, None)
    if n_page > cap_page:
        return (cap_page, None)
    return (n_page, n_y)


def split_into_cycles(q_subs, start_num: int):
    """Recover questions whose marker OCR missed but whose ``(a)/(b)/(c)``
    sub-parts were detected.  Each fresh ``(a)`` after the first sub-part
    starts a new virtual question number."""
    if not q_subs:
        return []
    groups = [(start_num, [q_subs[0]])]
    current_num = start_num
    for sp in q_subs[1:]:
        letter = sp[2]
        prev_letter = groups[-1][1][-1][2]
        if letter == "a":
            current_num += 1
            groups.append((current_num, [sp]))
        elif letter > prev_letter:
            groups[-1][1].append(sp)
    return groups


def subparts_in_question(subparts, q_start, q_end, page_height: int):
    s_page, s_y = q_start
    e_page, e_y = q_end
    same_line_tol = int(page_height * SAME_LINE_TOL_FRAC)
    out = []
    for sp in subparts:
        p, y, _letter = sp
        if p < s_page or p > e_page:
            continue
        if p == s_page and y < s_y - same_line_tol:
            continue
        if p == e_page and e_y is not None and y >= e_y - 4:
            continue
        out.append(sp)
    return out


def run_tesseract(pdf_path: Path, max_q: int) -> tuple[list[tuple[int, str | None, list[int]]], int]:
    """Run the Tesseract pipeline and return (per_page, n_pages)."""
    configure_tesseract()
    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    print(f"  {n_pages} pages, rendering at {DPI} DPI\n")
    print("Scanning pages for markers...")

    events, subparts, page_heights, page_year_raw = collect_markers(doc)
    doc.close()

    if not events:
        sys.exit(
            "\nNo question markers found.  Try lowering MIN_CONFIDENCE,\n"
            "raising DPI, or widening LEFT_MARGIN_*_FRAC."
        )

    # Carry years forward for cycle/region calculations
    page_years_carried: list[str | None] = []
    cur: str | None = None
    for y in page_year_raw:
        if y:
            cur = y
        page_years_carried.append(cur)

    # For each detected question marker, find its containing sub-parts and
    # use cycle inference to recover virtual questions.
    per_page_q: dict[int, set[int]] = {}
    for q_idx, (q_page, q_y, q_num, _tier) in enumerate(events):
        q_end = determine_question_end(events, q_idx, n_pages)
        year_tag = page_years_carried[q_page]
        if not year_tag:
            continue
        page_h = page_heights[q_page] or 3085
        q_subs = subparts_in_question(subparts, (q_page, q_y), q_end, page_h)
        if not q_subs:
            per_page_q.setdefault(q_page, set()).add(q_num)
            continue
        for c_idx, (v_num, v_subs) in enumerate(split_into_cycles(q_subs, q_num)):
            if not v_subs:
                continue
            page_for_q = q_page if c_idx == 0 else v_subs[0][0]
            per_page_q.setdefault(page_for_q, set()).add(v_num)

    print(f"\n{len(events)} question marker(s), {len(subparts)} sub-part(s) detected")
    detected_years = sorted({y for y in page_year_raw if y})
    print(f"  detected year tags: {detected_years or 'none'}\n")

    # Emit each year only on the page where it was first DETECTED (not
    # carried).  build_toc handles the carry-forward itself.
    seen_years: set[str] = set()
    per_page: list[tuple[int, str | None, list[int]]] = []
    for page_idx in range(n_pages):
        y_raw = page_year_raw[page_idx]
        emit_year: str | None = None
        if y_raw and y_raw not in seen_years:
            emit_year = y_raw
            seen_years.add(y_raw)
        q_nums = sorted(per_page_q.get(page_idx, set()))
        per_page.append((page_idx, emit_year, q_nums))
    return per_page, n_pages


# ═════════════════════════════════════════════════════════════════════════════
# Chandra (Datalab API) backend
# ═════════════════════════════════════════════════════════════════════════════


def split_paginated_markdown(md: str) -> list[str]:
    """Split Datalab/Marker's paginated markdown into per-page strings.

    Marker emits ``\\n\\n{n}-{48 dashes}\\n\\n`` between pages with
    ``paginate: true``.  We split on numeric + dashes lines and return
    the resulting page texts in order.
    """
    parts = PAGE_DELIM_RE.split(md)
    if len(parts) == 1:
        return [md]
    pages: list[str] = [parts[0]]
    # parts: [page0_text, num1, page1_text, num2, page2_text, ...]
    for i in range(2, len(parts), 2):
        pages.append(parts[i])
    return pages


def run_chandra(
    pdf_path: Path,
    max_q: int,
    api_key: str,
    use_force_ocr: bool = True,
) -> tuple[list[tuple[int, str | None, list[int]]], int]:
    """Submit ``pdf_path`` to the Datalab Chandra OCR API, poll until
    complete, and return (per_page, n_pages)."""
    try:
        import requests
    except ImportError:
        sys.exit("The 'requests' package is required for the Chandra "
                 "backend.  Run: pip install requests")
    import time

    if not api_key:
        sys.exit(
            "Datalab API key is required for the Chandra backend.\n"
            "Get one at https://www.datalab.to/app/api-keys and set\n"
            "the DATALAB_API_KEY environment variable, or pass --api-key."
        )

    headers = {"X-API-Key": api_key}
    submit_url = f"{DATALAB_API_BASE}/api/v1/marker"

    print(f"Submitting {pdf_path.name} to Datalab API ({submit_url})...")
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data = {
            "output_format": "markdown",
            "paginate": "true",
            "force_ocr": "true" if use_force_ocr else "false",
        }
        resp = requests.post(submit_url, files=files, data=data,
                             headers=headers, timeout=120)
    if resp.status_code >= 400:
        sys.exit(f"Submit failed: HTTP {resp.status_code} — {resp.text[:500]}")
    job = resp.json()
    check_url = job.get("request_check_url")
    if not check_url:
        rid = job.get("request_id") or job.get("id")
        if not rid:
            sys.exit(f"No request id in response: {job}")
        check_url = f"{DATALAB_API_BASE}/api/v1/marker/{rid}"

    print(f"  job submitted, polling every {DATALAB_POLL_INTERVAL_S}s "
          f"(max {DATALAB_MAX_WAIT_S}s)...")
    started = time.time()
    last_status = None
    while True:
        if time.time() - started > DATALAB_MAX_WAIT_S:
            sys.exit(f"Timed out after {DATALAB_MAX_WAIT_S}s waiting for job.")
        time.sleep(DATALAB_POLL_INTERVAL_S)
        r = requests.get(check_url, headers=headers, timeout=60)
        if r.status_code >= 400:
            sys.exit(f"Poll failed: HTTP {r.status_code} — {r.text[:500]}")
        result = r.json()
        status = result.get("status")
        if status != last_status:
            print(f"  status: {status}")
            last_status = status
        if status in ("complete", "completed"):
            break
        if status in ("failed", "error"):
            sys.exit(f"Datalab job failed: {result.get('error', result)}")

    md = result.get("markdown")
    if not md:
        sys.exit(f"No markdown in response.  Keys present: {list(result.keys())}")
    page_count = result.get("page_count")
    print(f"  done; {page_count or '?'} pages returned\n")

    pages = split_paginated_markdown(md)
    n_pages = max(len(pages), page_count or len(pages))
    print(f"Scanning {len(pages)} pages for year / question markers...")

    per_page: list[tuple[int, str | None, list[int]]] = []
    for page_idx, text in enumerate(pages):
        year = parse_year_string(text)
        q_nums = find_questions_in_markdown(text, max_q)
        per_page.append((page_idx, year, q_nums))
        if year or q_nums:
            qs = ",".join(str(n) for n in q_nums) or "-"
            year_tag = f"  [year {year}]" if year else ""
            print(f"  page {page_idx + 1}:  Q[{qs}]{year_tag}")

    return per_page, n_pages


# ═════════════════════════════════════════════════════════════════════════════
# CLI / main
# ═════════════════════════════════════════════════════════════════════════════


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add bookmark outlines (year + Q1-Q8) to a scanned exam-paper "
            "PDF.  Choose --backend tesseract (local, free) or --backend "
            "chandra (Datalab API, ~$0.005/page, much higher accuracy)."
        ),
    )
    p.add_argument("pdf", nargs="?", default=PDF_PATH,
                   help=f"Input PDF.  Default: {PDF_PATH!r}")
    p.add_argument("output", nargs="?", default=OUTPUT_PDF,
                   help=f"Output PDF path.  Default: {OUTPUT_PDF!r}")
    p.add_argument("--backend", choices=["tesseract", "chandra"],
                   default=DEFAULT_BACKEND,
                   help=f"OCR backend.  Default: {DEFAULT_BACKEND}")
    p.add_argument("--api-key", default=None,
                   help="Datalab API key (chandra backend).  Falls back to "
                        "the DATALAB_API_KEY env var if omitted.")
    p.add_argument("--max-q", type=int, default=MAX_QUESTION_FOR_OUTLINE,
                   help=f"Highest question number to outline.  "
                        f"Default: {MAX_QUESTION_FOR_OUTLINE}")
    p.add_argument("--dpi", type=int, default=DPI,
                   help=f"Render DPI for tesseract backend.  Default: {DPI}")
    p.add_argument("--skip", type=int, nargs="*", default=SKIP_PAGES,
                   help="1-indexed pages to ignore (tesseract backend).")
    p.add_argument("--no-force-ocr", action="store_true",
                   help="For chandra backend: let Datalab decide whether to "
                        "OCR (default forces OCR — your input is a scanned "
                        "PDF).")
    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    global DPI, SKIP_PAGES
    DPI = args.dpi
    SKIP_PAGES = list(args.skip)

    pdf_path = Path(args.pdf)
    out_path = Path(args.output)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")
    if out_path.resolve() == pdf_path.resolve():
        sys.exit("Output path must differ from input PDF path.")

    print(f"Opening {pdf_path.name}  (backend: {args.backend})")

    if args.backend == "chandra":
        api_key = args.api_key or os.environ.get("DATALAB_API_KEY", "")
        per_page, n_pages = run_chandra(
            pdf_path, args.max_q, api_key,
            use_force_ocr=not args.no_force_ocr,
        )
    else:
        per_page, n_pages = run_tesseract(pdf_path, args.max_q)

    toc = build_toc(per_page, args.max_q)
    if not toc:
        sys.exit("\nNo outline entries could be derived.")

    print(f"\nWriting outlined PDF...")
    doc = fitz.open(str(pdf_path))
    pdf_pages = len(doc)
    for entry in toc:
        if entry[2] > pdf_pages:
            print(f"  warning: clamping page {entry[2]} -> {pdf_pages} "
                  f"for entry {entry[1]!r}")
            entry[2] = pdf_pages
        elif entry[2] < 1:
            entry[2] = 1
    doc.set_toc(toc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()

    print(f"Wrote outlined PDF to:\n  {out_path}\n")
    print("Outline:")
    for level, title, page in toc:
        indent = "  " * (level - 1)
        print(f"  {indent}p{page}: {title}")


if __name__ == "__main__":
    main()
