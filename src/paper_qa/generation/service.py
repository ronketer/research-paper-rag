from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from paper_qa.config.model_config import create_chat_model
from paper_qa.generation.prompts import (
    COMPARISON_PROMPT,
    QA_PROMPT,
)
from paper_qa.retrieval.service import (
    retrieve_for_comparison,
    retrieve_for_question,
)


def get_llm():
    """Initialize the configured chat model."""

    return create_chat_model()


def build_qa_chain():
    """Build the single-paper QA generation chain."""

    return (
        QA_PROMPT
        | get_llm()
        | StrOutputParser()
    )


def _format_context_with_metadata(
    docs: list[Document],
) -> str:
    """Format retrieved evidence for model context."""

    chunks: list[str] = []

    for doc in docs:
        page = doc.metadata.get(
            "page_number",
            "Unknown",
        )

        section = doc.metadata.get(
            "section",
            "Unknown",
        )

        chunks.append(
            f"[Page {page} | Section: {section}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(chunks)


def _format_sources(
    docs: list[Document],
    fallback_paper_title: str | None,
) -> list[dict[str, Any]]:
    """Convert retrieved documents into UI-neutral source records."""

    return [
        {
            "text": doc.page_content,
            "paper": doc.metadata.get(
                "paper_title",
                fallback_paper_title or "Unknown",
            ),
            "page": doc.metadata.get(
                "page_number",
                "Unknown",
            ),
            "section": doc.metadata.get(
                "section",
                "Unknown",
            ),
        }
        for doc in docs
    ]


def answer_question(
    question: str,
    paper_title: str | None,
) -> dict[str, Any]:
    """Generate an answer from retrieved paper evidence."""

    docs = retrieve_for_question(
        question=question,
        paper_title=paper_title,
    )

    answer = build_qa_chain().invoke(
        {
            "context": _format_context_with_metadata(
                docs
            ),
            "question": question,
        }
    )

    return {
        "answer": answer,
        "sources": _format_sources(
            docs,
            paper_title,
        ),
    }


def compare_papers(
    question: str,
    paper_a: str,
    paper_b: str,
) -> dict[str, Any]:
    """Generate a comparison from evidence for two papers."""

    docs_a, docs_b = retrieve_for_comparison(
        question=question,
        paper_a=paper_a,
        paper_b=paper_b,
    )

    comparison_chain = (
        COMPARISON_PROMPT
        | get_llm()
        | StrOutputParser()
    )

    comparison = comparison_chain.invoke(
        {
            "paper_a": paper_a,
            "paper_b": paper_b,
            "context_a": _format_context_with_metadata(
                docs_a
            ),
            "context_b": _format_context_with_metadata(
                docs_b
            ),
            "question": question,
        }
    )

    return {
        "comparison": comparison,
        "sources_a": _format_sources(
            docs_a,
            paper_a,
        ),
        "sources_b": _format_sources(
            docs_b,
            paper_b,
        ),
    }
