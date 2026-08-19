from pathlib import Path

import pymupdf

from paper_qa.ingestion.models import Page


def load_pdf(file_path: str) -> list[Page]:
    """Extract non-empty PDF pages and their provenance metadata."""

    doc = pymupdf.open(file_path)

    try:
        title = (
            doc.metadata.get("title", "").strip()
            if doc.metadata
            else ""
        )

        if not title:
            title = (
                Path(file_path)
                .stem
                .replace("-", " ")
                .replace("_", " ")
                .title()
            )

        pages: list[Page] = []

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()

            if not text.strip():
                continue

            pages.append(
                Page(
                    page_number=page_number,
                    text=text,
                    paper_title=title,
                )
            )

        return pages

    finally:
        doc.close()


def load_multiple_pdfs(
    file_paths: list[str],
) -> dict[str, list[Page]]:
    """Load multiple PDFs keyed by resolved paper title."""

    result: dict[str, list[Page]] = {}

    for path in file_paths:
        pages = load_pdf(path)

        if pages:
            result[pages[0].paper_title] = pages

    return result
