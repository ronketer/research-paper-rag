import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_qa.ingestion.models import Chunk, Page

SECTION_PATTERNS = [
    r"^#{1,3}\s+.+",
    r"^\\(?:sub)*section\{.+\}",
    r"^\d+\.(?:\d+\.?)*\s+[A-Z].+",
    r"^(?:Abstract|Introduction|Conclusion|References|Appendix)\s*$",
]


def _is_markdown_heading_run(
    lines: list[str],
    line_index: int,
) -> bool:
    """Return whether a line belongs to a table/figure-like heading run."""

    markdown_heading = re.compile(r"^#{1,3}\s+.+")

    start = line_index

    while (
        start > 0
        and markdown_heading.match(lines[start - 1].strip())
    ):
        start -= 1

    end = line_index

    while (
        end + 1 < len(lines)
        and markdown_heading.match(lines[end + 1].strip())
    ):
        end += 1

    return end - start + 1 >= 3


def detect_section_boundaries(
    text: str,
) -> list[tuple[int, str]]:
    """Find line positions where new sections begin."""

    lines = text.splitlines()
    boundaries: list[tuple[int, str]] = []

    for index, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            continue

        for pattern in SECTION_PATTERNS:
            if not re.match(pattern, stripped):
                continue

            if (
                pattern == SECTION_PATTERNS[0]
                and _is_markdown_heading_run(lines, index)
            ):
                break

            title = re.sub(
                r"^#+\s+",
                "",
                stripped,
            )

            boundaries.append((index, title))
            break

    return boundaries


def _find_page_for_position(
    text: str,
    char_offset: int,
    pages: list[Page],
) -> int:
    """Resolve a concatenated-text position back to its source page."""

    last_page = (
        pages[0].page_number
        if pages
        else 1
    )

    for match in re.finditer(r"\[PAGE (\d+)\]", text):
        if match.start() <= char_offset:
            last_page = int(match.group(1))
        else:
            break

    return last_page


def naive_chunk(
    pages: list[Page],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    """Baseline recursive text chunker."""

    if not pages:
        return []

    paper_title = pages[0].paper_title

    full_text = "\n".join(
        f"[PAGE {page.page_number}]\n{page.text}"
        for page in pages
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    texts = splitter.split_text(full_text)

    chunks: list[Chunk] = []

    for index, text in enumerate(texts):
        position = full_text.find(text)

        page_number = _find_page_for_position(
            full_text,
            position,
            pages,
        )

        chunks.append(
            Chunk(
                text=text,
                paper_title=paper_title,
                section="Unknown",
                page_number=page_number,
                chunk_index=index,
            )
        )

    return chunks


def section_aware_chunk(
    pages: list[Page],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    """Split by detected sections before applying size-based splitting."""

    if not pages:
        return []

    paper_title = pages[0].paper_title

    full_text = "\n".join(
        f"[PAGE {page.page_number}]\n{page.text}"
        for page in pages
    )

    lines = full_text.splitlines()
    boundaries = detect_section_boundaries(full_text)

    sections: list[tuple[str, str]] = []

    if not boundaries:
        sections.append(
            ("Full Document", full_text)
        )

    else:
        if boundaries[0][0] > 0:
            preamble = "\n".join(
                lines[:boundaries[0][0]]
            )

            if preamble.strip():
                sections.append(
                    ("Preamble", preamble)
                )

        for index, (line_index, title) in enumerate(
            boundaries
        ):
            if index + 1 < len(boundaries):
                end_index = boundaries[index + 1][0]
            else:
                end_index = len(lines)

            section_text = "\n".join(
                lines[line_index:end_index]
            )

            if section_text.strip():
                sections.append(
                    (title, section_text)
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks: list[Chunk] = []
    chunk_index = 0

    for title, section_text in sections:
        if len(section_text) > chunk_size:
            sub_texts = splitter.split_text(
                section_text
            )
        else:
            sub_texts = [section_text]

        for sub_text in sub_texts:
            position = full_text.find(sub_text)

            page_number = _find_page_for_position(
                full_text,
                position,
                pages,
            )

            chunks.append(
                Chunk(
                    text=sub_text,
                    paper_title=paper_title,
                    section=title,
                    page_number=page_number,
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

    return chunks
