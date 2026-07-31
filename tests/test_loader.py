"""Tests for src/loader.py"""

import pytest
from src.loader import load_pdf, load_multiple_pdfs, Page

# ---------------------------------------------------------------------------
# load_pdf — basic behavior
# ---------------------------------------------------------------------------

def test_load_pdf_returns_list_of_pages(sample_pdf):
    """load_pdf should return a non-empty list of Page objects."""
    pages = load_pdf(sample_pdf)

    assert isinstance(pages, list)
    assert len(pages) > 0
    assert all(isinstance(p, Page) for p in pages)


def test_page_has_correct_fields(sample_pdf):
    """Each Page should have page_number, text, and paper_title filled in."""
    page = load_pdf(sample_pdf)[0]

    assert isinstance(page.page_number, int)
    assert isinstance(page.text, str)
    assert isinstance(page.paper_title, str)
    assert len(page.text) > 0
    assert len(page.paper_title) > 0


def test_page_numbers_start_at_one(sample_pdf):
    """Page numbering should be 1-indexed."""
    pages = load_pdf(sample_pdf)
    assert pages[0].page_number == 1


def test_page_numbers_are_ascending(sample_pdf):
    """Page numbers should increase (but may skip if blank pages are dropped)."""
    pages = load_pdf(sample_pdf)
    for i in range(1, len(pages)):
        assert pages[i].page_number > pages[i - 1].page_number


def test_all_pages_share_same_title(sample_pdf):
    """Every page from one PDF should have the same paper_title."""
    pages = load_pdf(sample_pdf)
    titles = {p.paper_title for p in pages}
    assert len(titles) == 1


# ---------------------------------------------------------------------------
# load_pdf — edge cases
# ---------------------------------------------------------------------------

def test_load_pdf_nonexistent_file_raises():
    """Passing a path that doesn't exist should raise an error."""
    with pytest.raises(Exception):
        load_pdf("this_file_does_not_exist.pdf")


def test_blank_pages_are_skipped(sample_pdf):
    """Pages with no text content should not appear in the result."""
    pages = load_pdf(sample_pdf)
    for page in pages:
        assert page.text.strip() != ""


# ---------------------------------------------------------------------------
# load_multiple_pdfs
# ---------------------------------------------------------------------------

def test_load_multiple_returns_dict(papers_dir):
    """load_multiple_pdfs should return a dict keyed by title."""
    pdfs = [str(p) for p in sorted(papers_dir.glob("*.pdf"))[:1]]
    result = load_multiple_pdfs(pdfs)

    assert isinstance(result, dict)
    assert len(result) == 1

    # The single key should be the paper title
    title = list(result.keys())[0]
    assert isinstance(title, str)
    assert len(title) > 0

    # The value should be a list of Pages
    pages = result[title]
    assert isinstance(pages, list)
    assert all(isinstance(p, Page) for p in pages)


def test_load_multiple_all_papers(papers_dir):
    """load_multiple_pdfs on the full papers/ folder returns one entry per PDF."""
    pdfs = [str(p) for p in sorted(papers_dir.glob("*.pdf"))]
    result = load_multiple_pdfs(pdfs)

    assert len(result) == len(pdfs)
    for title, pages in result.items():
        assert isinstance(pages, list)
        assert len(pages) > 0
