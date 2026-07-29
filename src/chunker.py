
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loader import Page


@dataclass
class Chunk:
    """A chunk of text with metadata for retrieval."""
    text: str              # the chunk content
    paper_title: str       # which paper this came from
    section: str           # e.g. "3.2 Multi-Head Attention"
    page_number: int       # page where this chunk starts
    chunk_index: int       # position within the paper




SECTION_PATTERNS = [
    r"^#{1,3}\s+.+",                    # Markdown: ## Abstract
    r"^\\(?:sub)*section\{.+\}",        # LaTeX: \section{Introduction}
    r"^\d+\.(?:\d+\.?)*\s+[A-Z].+",    # Numbered: 3.2 Attention Mechanism
    r"^(?:Abstract|Introduction|Conclusion|References|Appendix)\s*$",
]


def detect_section_boundaries(text: str) -> list[tuple[int, str]]:
    """
    Find line positions where new sections begin.

    Returns:
        List of (line_index, section_title) tuples
    """
    lines = text.splitlines()
    boundaries = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in SECTION_PATTERNS:
            if re.match(pattern, stripped):
                # Clean up markdown heading markers for the title
                title = re.sub(r"^#+\s+", "", stripped)
                boundaries.append((i, title))
                break
    return boundaries


def _find_page_for_position(text: str, char_offset: int, pages: list[Page]) -> int:
    """
    Given a character offset into the full concatenated text,
    determine which page number it falls on by finding the last
    [PAGE N] marker at or before that offset.
    """
    last_page = pages[0].page_number if pages else 1
    for match in re.finditer(r"\[PAGE (\d+)\]", text):
        if match.start() <= char_offset:
            last_page = int(match.group(1))
        else:
            break
    return last_page



def naive_chunk(pages: list[Page], chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """
    Baseline chunker using LangChain's RecursiveCharacterTextSplitter.
    """
    if not pages:
        return []

    paper_title = pages[0].paper_title

    # Concatenate with page markers so we can recover page numbers
    full_text = "\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    texts = splitter.split_text(full_text)

    chunks = []
    for i, text in enumerate(texts):
        page_num = _find_page_for_position(full_text, full_text.find(text), pages)
        chunks.append(
            Chunk(
                text=text,
                paper_title=paper_title,
                section="Unknown",
                page_number=page_num,
                chunk_index=i,
            )
        )
    return chunks


def section_aware_chunk(pages: list[Page], chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """
    Splits text at section boundaries first,
    then applies size-based splitting within sections.

    Key logic:
    1. Concatenate all pages into one text (preserving page markers)
    2. Detect section boundaries
    3. Split into sections first
    4. If a section is too long, split it further (but never mid-equation)
    5. Attach metadata: paper_title, section name, page_number
    """
    if not pages:
        return []

    paper_title = pages[0].paper_title

    # Step 1: Concatenate all pages into one text with page markers
    full_text = "\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in pages)
    lines = full_text.splitlines()

    # Step 2: Detect section boundaries
    boundaries = detect_section_boundaries(full_text)

    # Step 3: Split into sections
    # Each section spans from its boundary line to the next boundary (exclusive)
    sections: list[tuple[str, str]] = []

    if not boundaries:
        # No section headings found — treat entire text as one section
        sections.append(("Full Document", full_text))
    else:
        # Text before the first section heading
        if boundaries[0][0] > 0:
            pre_text = "\n".join(lines[:boundaries[0][0]])
            if pre_text.strip():
                sections.append(("Preamble", pre_text))

        # Each detected section
        for idx, (line_idx, title) in enumerate(boundaries):
            if idx + 1 < len(boundaries):
                end_idx = boundaries[idx + 1][0]
            else:
                end_idx = len(lines)
            section_text = "\n".join(lines[line_idx:end_idx])
            if section_text.strip():
                sections.append((title, section_text))

    # Step 4: Split sections further if too long, attach metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks: list[Chunk] = []
    global_chunk_index = 0

    for title, section_text in sections:
        if len(section_text) > chunk_size:
            sub_texts = splitter.split_text(section_text)
        else:
            sub_texts = [section_text]

        for sub_text in sub_texts:
            # Find the page number by locating this text's position in the full document
            pos = full_text.find(sub_text)
            page_num = _find_page_for_position(full_text, pos, pages)

            chunks.append(
                Chunk(
                    text=sub_text,
                    paper_title=paper_title,
                    section=title,
                    page_number=page_num,
                    chunk_index=global_chunk_index,
                )
            )
            global_chunk_index += 1

    return chunks
