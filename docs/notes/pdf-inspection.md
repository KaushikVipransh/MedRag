# PDF inspection — `data/sources/dhingra-ent-7e.pdf`

Generated with `docker compose run --rm dev python scripts/inspect_pdf.py` on 2026-10-11.

## Page count / text pages
- 556 PDF pages; **0 pages without a text layer** (no OCR needed).
- 53 pages are "Page left intentionally blank" (dropped by the <100-char filter).
- Total extracted text ≈ 1.61 M characters → **≈ 2,000 chunks** expected at ~200 tokens per chunk. The v1 paper's "634 chunks" does not match this book at 700 characters per chunk, which would give ≈ 2,300+.

## Embedded page labels
- Absent (`get_label()` returns `""` on every page) → labels must be inferred from footer/header numbers.

## Outline (TOC)
- 2,744 entries across levels 1–7.
- **Levels ≥ 3 are broken:** they all point to page 1 (e.g. `[3, 'The External Ear', 1]`).
- **Levels 1–2 are clean:** 121 entries, all with valid pages, in increasing order (Section → Chapter, plus front matter, appendices and index). → Sections come from the TOC with `max_level=2`, giving `"Section I - Diseases of Ear > Chapter 1 - Anatomy of Ear"`.
- Some titles contain non-breaking spaces (`\xa0`), which need whitespace normalisation.

## Printed-page offset
- Votes `(printed − pdf_index)`: **−14 → 481 votes**; the next best has 2. The offset is dominant.
- Spot check: PDF index 160 shows printed page **146** (= 160 − 14). ✔
- **Consequence for v1:** v1 cited `pdf_index + 1`, so every v1 citation was 15 pages too high relative to the printed book (v1 "Page 160" for tinnitus = printed page 146).

## Running headers / footers (normalised margin lines)
| count | line |
|---|---|
| 550 | `mebooksfree.com` (watermark on nearly every page) |
| 482 | `#` (bare page number) |
| 83 | `c h a p t e r # #` (letter-spaced chapter header) |
| 64 | `section i — diseases of ear` |
| 53 | `page left intentionally blank` |
| 38, 24, 18, … | other section running headers |

All of these exceed the default `min_count=4`, so `find_repeating_margin_lines` catches them.

## Text quirks that cleaning must handle
- **Ligatures:** `ﬁ` (U+FB01) in words like `signiﬁcant` and `superﬁcial` → NFKC normalisation is required, or BM25 and embeddings will miss these words.
- **Bullet glyph followed by U+0E00** (an unassigned code point), e.g. `•฀Lymphoma` → unassigned characters must become spaces so bullets are detected.
- Hyphenated line breaks (`schwanno-` / `ma`), which `lines_to_text` already rejoins.

## Source note
The `mebooksfree.com` watermark shows this PDF was obtained from a third-party download site. That reinforces the spec's rule that the PDF and anything derived from it (the index) are never published.
