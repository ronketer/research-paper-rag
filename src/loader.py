import pymupdf  
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    """A single page extracted from a PDF."""
    page_number: int       # 1-indexed
    text: str              # raw text content of the page
    paper_title: str       # extracted from PDF metadata or filename


def load_pdf(file_path: str) -> list[Page]:
    """
    Load a PDF and extract text page by page.

    Args:
        file_path: path to the PDF file

    Returns:
        List of Page objects with text and page numbers
    """
    # 1. Open PDF with PyMuPDF
    doc = pymupdf.open(file_path)

    # 2. Extract metadata (title) or fall back to filename
    title = doc.metadata.get("title", "").strip() if doc.metadata else ""
    if not title:
        title = Path(file_path).stem.replace("-", " ").replace("_", " ").title()

    # 3. Loop through pages, extract text
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():  # skip completely blank pages
            pages.append(
                Page(
                    page_number=page_num,
                    text=text,
                    paper_title=title,
                )
            )

    doc.close()
    return pages


def load_multiple_pdfs(file_paths: list[str]) -> dict[str, list[Page]]:
    """Load multiple PDFs, keyed by paper title."""
    result = {}
    for path in file_paths:
        pages = load_pdf(path)
        if pages:
            title = pages[0].paper_title
            result[title] = pages
    return result