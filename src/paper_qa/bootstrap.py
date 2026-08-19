"""Application composition root.

This module is the one place where abstract application workflows are
connected to concrete loaders, stores, retrieval/generation code, and routing.
"""

from paper_qa.application.rag_service import (
    IngestionResult,
    QueryResult,
    RAGService,
)
from paper_qa.generation.service import (
    answer_question,
    compare_papers,
)
from paper_qa.infrastructure.vectorstore import (
    add_paper,
    delete_paper,
    list_papers,
)
from paper_qa.ingestion.chunker import (
    naive_chunk,
    section_aware_chunk,
)
from paper_qa.ingestion.loader import load_pdf
from paper_qa.routing.router import classify_query


def create_rag_service() -> RAGService:
    """Wire application workflows to concrete implementations."""

    return RAGService(
        load_pdf_fn=load_pdf,
        chunkers={
            "naive": naive_chunk,
            "section_aware": section_aware_chunk,
        },
        add_paper_fn=add_paper,
        delete_paper_fn=delete_paper,
        list_papers_fn=list_papers,
        answer_question_fn=answer_question,
        compare_papers_fn=compare_papers,
        classify_query_fn=classify_query,
    )


_service = create_rag_service()


def ingest_document(
    file_path: str,
    chunking_strategy: str,
) -> IngestionResult:
    """Ingest a document through the application service."""

    return _service.ingest_document(
        file_path=file_path,
        chunking_strategy=chunking_strategy,
    )


def run_query(
    question: str,
    selected_papers: list[str] | None = None,
) -> QueryResult:
    """Execute a query through the application service."""

    return _service.run_query(
        question=question,
        selected_papers=selected_papers,
    )


def list_ingested_papers() -> list[str]:
    """Return currently ingested papers."""

    return _service.list_papers()


def delete_ingested_paper(
    paper_title: str,
) -> None:
    """Delete one ingested paper."""

    _service.delete_paper(paper_title)
