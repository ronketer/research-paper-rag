from langchain_core.documents import Document

from paper_qa.infrastructure.vectorstore import retrieve

QUESTION_RETRIEVAL_K = 4
COMPARISON_RETRIEVAL_K = 3


def retrieve_for_question(
    question: str,
    paper_title: str | None,
) -> list[Document]:
    """Apply retrieval policy for a Q&A request."""

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
    """Retrieve balanced evidence for two papers."""

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
