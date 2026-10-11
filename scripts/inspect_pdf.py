"""Print structural facts about a PDF that ingestion depends on."""
import re
import sys
from collections import Counter

import pymupdf

path = sys.argv[1] if len(sys.argv) > 1 else "data/sources/dhingra-ent-7e.pdf"
doc = pymupdf.open(path)
pages = [p.get_text("text").splitlines() for p in doc]

print(f"file: {path}")
print(f"pages: {len(doc)}")
print(f"metadata title: {doc.metadata.get('title')!r}")

labels = [p.get_label() for p in doc]
print(f"embedded page labels present: {any(labels)}; sample: {labels[:3]} ... {labels[100:103]}")

toc = doc.get_toc()
print(f"outline entries: {len(toc)}; levels: {Counter(e[0] for e in toc)}")
for entry in toc[:15]:
    print("   ", entry)

no_text = [i for i, lines in enumerate(pages) if not "".join(lines).strip()]
print(f"pages without text: {len(no_text)} {no_text[:20]}")

total_chars = sum(len("\n".join(lines)) for lines in pages)
print(f"total chars: {total_chars:,}  (≈ {total_chars // 800:,} chunks of ~200 tokens)")

norm = lambda s: re.sub(r"\d+", "#", s.strip().lower())
margin = Counter()
for lines in pages:
    margin.update({norm(l) for l in lines[:3] + lines[-3:] if l.strip()})
print("most common margin lines (normalised):")
for line, n in margin.most_common(15):
    print(f"   {n:4d}  {line[:70]!r}")

votes = Counter()
num = re.compile(r"^(\d{1,4})(?:\s|$)|\s(\d{1,4})$")
for i, lines in enumerate(pages):
    for l in lines[:3] + lines[-3:]:
        m = num.search(l.strip())
        if m:
            votes[int(m.group(1) or m.group(2)) - i] += 1
print(f"printed-page offset votes (printed - pdf_index): {votes.most_common(5)}")

for i in (20, len(doc) // 4, len(doc) // 2):
    print(f"--- pdf index {i}: first 4 lines {pages[i][:4]} | last 3 lines {pages[i][-3:]}")
