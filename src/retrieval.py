"""Retrieval policies for RAG question answering and comparison."""

from langchain_core.documents import Document

from src.vectorstore import retrieve

QUESTION_RETRIEVAL_K = 4
COMPARISON_RETRIEVAL_K = 3


def retrieve_for_question(
    question: str,
    paper_title: str | None,
) -> list[Document]:
    """Retrieve evidence for a single-paper or all-paper question.

    The retrieval policy lives here so generation code does not need to know
    how many chunks are fetched or how the vector store is queried.
    """
    return retrieve(
        query=question,
        paper_filter=paper_title,
        k=QUESTION_RETRIEVAL_K,
    )


def retrieve_for_comparison(
    question: str,
    paper_a: str,
    paper_b: str,
) -> tuple[list[Document], list[Document]]:
    """Retrieve evidence independently for two papers being compared."""
    docs_a = retrieve(
        query=question,
        paper_filter=paper_a,
        k=COMPARISON_RETRIEVAL_K,
    )

    docs_b = retrieve(
        query=question,
        paper_filter=paper_b,
        k=COMPARISON_RETRIEVAL_K,
    )

    return docs_a, docs_b
