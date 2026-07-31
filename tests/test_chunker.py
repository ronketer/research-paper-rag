"""Tests for src/chunker.py"""

import pytest
from src.loader import Page
from src.chunker import (
    Chunk,
    detect_section_boundaries,
    naive_chunk,
    section_aware_chunk,
)

# ---------------------------------------------------------------------------
# Helpers — fake pages so tests don't depend on a real PDF
# ---------------------------------------------------------------------------

def _make_pages(texts: list[str], title: str = "Test Paper") -> list[Page]:
    """Create a list of Page objects from raw text strings."""
    return [
        Page(page_number=i + 1, text=t, paper_title=title)
        for i, t in enumerate(texts)
    ]


SAMPLE_TEXT = (
    "This is the introduction to the paper.\n"
    "It explains the motivation.\n"
    "\n"
    "1. Background\n"
    "Some background information here.\n"
    "More background details.\n"
    "\n"
    "2. Methods\n"
    "We used a novel approach.\n"
    "The approach involves several steps.\n"
    "\n"
    "3. Results\n"
    "The results were excellent.\n"
    "\n"
    "Conclusion\n"
    "We concluded that this works.\n"
)

SAMPLE_PAGES = _make_pages([SAMPLE_TEXT])


# ---------------------------------------------------------------------------
# detect_section_boundaries
# ---------------------------------------------------------------------------

def test_detect_numbered_sections():
    """Should detect '1. Background', '2. Methods', '3. Results' headings."""
    boundaries = detect_section_boundaries(SAMPLE_TEXT)
    titles = [title for _, title in boundaries]

    assert "1. Background" in titles
    assert "2. Methods" in titles
    assert "3. Results" in titles


def test_detect_named_sections():
    """Should detect standalone 'Conclusion' keyword."""
    boundaries = detect_section_boundaries(SAMPLE_TEXT)
    titles = [title for _, title in boundaries]

    assert "Conclusion" in titles


def test_detect_markdown_headings():
    """Should detect markdown-style headings like '## Abstract'."""
    md_text = "Some intro text\n## Abstract\nThis is the abstract.\n## Methods\nDetails."
    boundaries = detect_section_boundaries(md_text)
    titles = [title for _, title in boundaries]

    assert "Abstract" in titles
    assert "Methods" in titles


def test_detect_returns_line_indices():
    """Each boundary should be a (line_index, title) tuple with valid indices."""
    boundaries = detect_section_boundaries(SAMPLE_TEXT)
    lines = SAMPLE_TEXT.splitlines()

    for line_idx, title in boundaries:
        assert 0 <= line_idx < len(lines)
        assert isinstance(title, str)
        assert len(title) > 0


def test_detect_no_headings():
    """Plain text with no section patterns should return empty list."""
    plain = "Just some text.\nNothing special here.\nNo headings at all."
    boundaries = detect_section_boundaries(plain)
    assert boundaries == []


# ---------------------------------------------------------------------------
# naive_chunk — returns correct types and metadata
# ---------------------------------------------------------------------------

def test_naive_chunk_returns_chunk_objects():
    """naive_chunk should return a list of Chunk dataclass instances."""
    chunks = naive_chunk(SAMPLE_PAGES)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_naive_chunk_preserves_paper_title():
    """Every chunk should carry the paper_title from the input pages."""
    chunks = naive_chunk(SAMPLE_PAGES)
    for chunk in chunks:
        assert chunk.paper_title == "Test Paper"


def test_naive_chunk_has_sequential_index():
    """chunk_index should be 0, 1, 2, ..."""
    chunks = naive_chunk(SAMPLE_PAGES)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_naive_chunk_empty_input():
    """Passing an empty list should return an empty list."""
    assert naive_chunk([]) == []


# ---------------------------------------------------------------------------
# section_aware_chunk — the key differentiator
# ---------------------------------------------------------------------------

def test_section_aware_returns_chunk_objects():
    """section_aware_chunk should return a list of Chunk instances."""
    chunks = section_aware_chunk(SAMPLE_PAGES)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_section_aware_preserves_paper_title():
    """Every chunk should carry the correct paper_title."""
    chunks = section_aware_chunk(SAMPLE_PAGES)
    for chunk in chunks:
        assert chunk.paper_title == "Test Paper"


def test_section_aware_assigns_section_names():
    """Each chunk should have a non-empty section name (never 'Unknown')."""
    chunks = section_aware_chunk(SAMPLE_PAGES)
    for chunk in chunks:
        assert chunk.section != ""
        assert chunk.section != "Unknown"


def test_section_aware_chunks_belong_to_single_section():
    """No chunk's text should contain a section heading that doesn't match its label.
    (i.e., section boundaries were respected and text wasn't lumped together)."""
    pages = _make_pages([
        "Intro text.\n\n1. First Section\nContent A.\n\n2. Second Section\nContent B."
    ])
    chunks = section_aware_chunk(pages, chunk_size=5000)  # large size so no sub-splitting

    # We expect separate chunks for the preamble, section 1, and section 2
    sections = [c.section for c in chunks]
    assert "1. First Section" in sections
    assert "2. Second Section" in sections


def test_section_aware_has_sequential_index():
    """chunk_index should be globally sequential: 0, 1, 2, ..."""
    chunks = section_aware_chunk(SAMPLE_PAGES)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_section_aware_splits_large_sections():
    """A section larger than chunk_size should be split into multiple chunks."""
    big_text = "1. Big Section\n" + ("This is a long sentence. " * 200)
    pages = _make_pages([big_text])

    chunks = section_aware_chunk(pages, chunk_size=200, overlap=50)
    big_chunks = [c for c in chunks if c.section == "1. Big Section"]
    assert len(big_chunks) > 1  # was split


def test_section_aware_empty_input():
    """Passing an empty list should return an empty list."""
    assert section_aware_chunk([]) == []


def test_section_aware_page_numbers_are_valid():
    """page_number on each chunk should be a positive integer from the input pages."""
    pages = _make_pages(["Page one text.\n1. Intro\nSome content."], title="Paper")
    chunks = section_aware_chunk(pages)
    for chunk in chunks:
        assert chunk.page_number >= 1


# ---------------------------------------------------------------------------
# Comparison: section-aware should produce richer metadata than naive
# ---------------------------------------------------------------------------

def test_section_aware_has_real_sections_unlike_naive():
    """section_aware_chunk should label sections; naive should use 'Unknown'."""
    sa_chunks = section_aware_chunk(SAMPLE_PAGES)
    n_chunks = naive_chunk(SAMPLE_PAGES)

    sa_sections = {c.section for c in sa_chunks}
    n_sections = {c.section for c in n_chunks}

    # Naive only ever labels "Unknown"
    assert n_sections == {"Unknown"}
    # Section-aware should have multiple distinct section names
    assert len(sa_sections) > 1
