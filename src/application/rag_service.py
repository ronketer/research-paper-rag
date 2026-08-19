from dataclasses import dataclass
from typing import Any

from src.chains import answer_question, compare_papers
from src.chunker import naive_chunk, section_aware_chunk
from src.loader import load_pdf
from src.router import classify_query
from src.vectorstore import add_paper, delete_paper, list_papers


@dataclass(frozen=True)
class IngestionResult:
    paper_title: str
    chunk_count: int


@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[dict[str, Any]]
    source_heading: str


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


def run_query(
    question: str,
    selected_papers: list[str] | None = None,
) -> QueryResult:
    """Execute single-paper Q&A or multi-paper comparison."""
    selected = selected_papers or []

    if len(selected) == 2:
        paper_a, paper_b = selected
        result = compare_papers(question, paper_a, paper_b)
        return QueryResult(
            answer=result["comparison"],
            sources=result["sources_a"] + result["sources_b"],
            source_heading=f"Comparison: {paper_a} vs {paper_b}",
        )

    if len(selected) == 1:
        paper = selected[0]
        result = answer_question(question, paper)
        return QueryResult(
            answer=result["answer"],
            sources=result["sources"],
            source_heading=f"Paper: {paper}",
        )

    available_papers = list_papers()
    route = classify_query(question, available_papers)
    if route["mode"] == "comparison" and len(route["papers"]) >= 2:
        paper_a, paper_b = route["papers"][:2]
        result = compare_papers(question, paper_a, paper_b)
        return QueryResult(
            answer=result["comparison"],
            sources=result["sources_a"] + result["sources_b"],
            source_heading=f"Comparison: {paper_a} vs {paper_b}",
        )

    paper = route["papers"][0] if route["papers"] else None
    result = answer_question(question, paper)
    return QueryResult(
        answer=result["answer"],
        sources=result["sources"],
        source_heading=f"Paper: {paper or 'all papers'}",
    )

