from collections import Counter


def sections_from_toc(toc: list[list], n_pages: int, max_level: int = 2) -> list[str]:
    marks: list[tuple[int, str]] = []
    path: list[str] = []
    for level, title, page in toc:
        if page < 1 or level > max_level:
            continue
        path = path[: level - 1] + [" ".join(title.split())]
        marks.append((page - 1, " > ".join(path)))
    marks.sort(key=lambda m: m[0])  # stable: keeps document order within a page

    result, current, k = [], "", 0
    for i in range(n_pages):
        while k < len(marks) and marks[k][0] <= i:
            current = marks[k][1]
            k += 1
        result.append(current)
    return result


def sections_from_fonts(
    font_lines: list[list[tuple[str, float]]], min_ratio: float = 1.3, max_len: int = 80
) -> list[str]:
    """Fallback for PDFs without an outline: lines set noticeably larger than the body
    font are treated as headings."""
    sizes: Counter[float] = Counter()
    for page in font_lines:
        for text, size in page:
            sizes[round(size, 1)] += len(text)
    if not sizes:
        return [""] * len(font_lines)
    body = sizes.most_common(1)[0][0]

    result, current = [], ""
    for page in font_lines:
        for text, size in page:
            if size >= body * min_ratio and len(text) <= max_len and any(c.isalpha() for c in text):
                current = text.strip()
        result.append(current)
    return result
