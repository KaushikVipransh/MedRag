# Ingest report — Dhingra ENT 7e

Run on 2026-10-11:

```
docker compose run --rm dev python -m medrag.ingest data/sources/dhingra-ent-7e.pdf --title "Dhingra - Diseases of Ear, Nose and Throat (7th ed.)"
dhingra-ent-7e: 2400 chunks from 480/556 pages in 41s
```

Settings: all-MiniLM-L6-v2, chunks ≤ 200 tokens, 40-token overlap, index on the Docker volume at `/var/medrag/chroma`.

## Checks

| # | Check | Result |
|---|---|---|
| 1 | PDF index 160 → printed page | `146`, matching the printed page seen in Task 2 ✔ (section: *Section I - Diseases of Ear > Chapter 22 - Tinnitus*) |
| 2 | Running headers/watermark/page numbers in chunk text | 0 chunks contain `mebooksfree` or `c h a p t e r` ✔. A sample of 6 random chunks was clean. |
| 3 | Sections | Every chunk has a section. Chapter-level paths such as `Section V - Diseases of Larynx and Trachea > Chapter 64 - Tracheostomy` ✔ |
| 4 | Ménière's disease | Chunks concentrate on printed pages 111–115 (Chapter on Ménière's disease) ✔. v1 cited "Page 126", which is PDF index 125, i.e. printed page 111. v1 was right about the location but 15 off in numbering. |
| 5 | Chunk count | **2,400** (mean 173 tokens, max 200, none over MiniLM's 256 limit). The v1 paper's "634 chunks" can't be reproduced from this book with 700-character chunks; it was either a different configuration or an error. |

## Known residue
- 12 front-matter chunks (Title Page 1, Copyright 4, Dedication 1, Preface 4, Acknowledgements 2), 0.5% of the index. They are left in for now; Plan 2 will measure whether they are ever retrieved.
- Index and Contents pages were correctly excluded (0 chunks).
- 76 pages yielded no chunks: blank "intentionally left blank" pages, the Index and Contents, and the cover/title pages.
