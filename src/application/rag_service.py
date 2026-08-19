from dataclasses import dataclass

from src.chunker import naive_chunk, section_aware_chunk
from src.loader import load_pdf
from src.vectorstore import add_paper, delete_paper


@dataclass(frozen=True)
class IngestionResult:
    paper_title: str
    chunk_count: int


CHUNKERS = {
    "naive": naive_chunk,
    "section_aware": section_aware_chunk,
}


def ingest_document(
    file_path: str,
    chunking_strategy: str,
) -> IngestionResult:
    """Load, chunk, and persist a PDF."""

    pages = load_pdf(file_path)

    if not pages:
        raise ValueError("No text could be extracted from this PDF.")

    chunker = CHUNKERS.get(chunking_strategy)

    if chunker is None:
        raise ValueError(
            f"Unknown chunking strategy: {chunking_strategy}"
        )

    title = pages[0].paper_title
    chunks = chunker(pages)

    # Replace an existing version of the same paper.
    delete_paper(title)
    add_paper(title, chunks)

    return IngestionResult(
        paper_title=title,
        chunk_count=len(chunks),
    )
