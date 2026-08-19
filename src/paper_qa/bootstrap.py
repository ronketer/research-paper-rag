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
    add_paper as store_add_paper,
)
from paper_qa.infrastructure.vectorstore import (
    delete_paper as store_delete_paper,
)
from paper_qa.infrastructure.vectorstore import (
    list_papers as store_list_papers,
)
from paper_qa.ingestion.chunker import (
    naive_chunk,
    section_aware_chunk,
)
from paper_qa.ingestion.loader import load_pdf
from paper_qa.routing.router import classify_query


def create_rag_service() -> RAGService:
    """Wire application use cases to concrete implementations."""

    return RAGService(
        load_pdf_fn=load_pdf,
        chunkers={
            "naive": naive_chunk,
            "section_aware": section_aware_chunk,
        },
        add_paper_fn=store_add_paper,
        delete_paper_fn=store_delete_paper,
        list_papers_fn=store_list_papers,
        answer_question_fn=answer_question,
        compare_papers_fn=compare_papers,
        classify_query_fn=classify_query,
    )


def ingest_document(
    file_path: str,
    chunking_strategy: str,
) -> IngestionResult:
    return create_rag_service().ingest_document(
        file_path=file_path,
        chunking_strategy=chunking_strategy,
    )


def run_query(
    question: str,
    selected_papers: list[str] | None = None,
) -> QueryResult:
    return create_rag_service().run_query(
        question=question,
        selected_papers=selected_papers,
    )


def list_papers() -> list[str]:
    return create_rag_service().list_papers()


def delete_paper(
    paper_title: str,
) -> None:
    create_rag_service().delete_paper(
        paper_title
    )
