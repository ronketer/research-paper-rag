"""LLM generation chains for paper Q&A and comparison."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.model_config import create_chat_model
from src.prompts import COMPARISON_PROMPT, QA_PROMPT
from src.retrieval import (
    retrieve_for_comparison,
    retrieve_for_question,
)


def get_llm():
    """Initialize the configured local or hosted chat model."""
    return create_chat_model()


def build_qa_chain():
    """Build a single-paper QA chain using LCEL."""
    llm = get_llm()
    output_parser = StrOutputParser()

    return QA_PROMPT.pipe(llm).pipe(output_parser)


def _format_context_with_metadata(
    docs: list[Document],
) -> str:
    """Format retrieved chunks with page and section metadata."""
    formatted_chunks = []

    for doc in docs:
        page = doc.metadata.get(
            "page_number",
            "Unknown",
        )
        section = doc.metadata.get(
            "section",
            "Unknown",
        )

        snippet = (
            f"[Page {page} | Section: {section}]\n"
            f"{doc.page_content}"
        )

        formatted_chunks.append(snippet)

    return "\n\n".join(formatted_chunks)


def _format_sources(
    docs: list[Document],
    fallback_paper_title: str | None,
) -> list[dict]:
    """Convert retrieved documents into source records for the UI."""
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
) -> dict:
    """Generate an answer from retrieved paper evidence."""
    docs = retrieve_for_question(
        question=question,
        paper_title=paper_title,
    )

    context_text = _format_context_with_metadata(docs)

    chain = build_qa_chain()

    answer = chain.invoke(
        {
            "context": context_text,
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
) -> dict:
    """Generate a comparison between two papers."""
    docs_a, docs_b = retrieve_for_comparison(
        question=question,
        paper_a=paper_a,
        paper_b=paper_b,
    )

    context_a_text = _format_context_with_metadata(
        docs_a
    )
    context_b_text = _format_context_with_metadata(
        docs_b
    )

    llm = get_llm()
    output_parser = StrOutputParser()

    comparison_chain = (
        COMPARISON_PROMPT
        | llm
        | output_parser
    )

    comparison_table = comparison_chain.invoke(
        {
            "paper_a": paper_a,
            "paper_b": paper_b,
            "context_a": context_a_text,
            "context_b": context_b_text,
            "question": question,
        }
    )

    return {
        "comparison": comparison_table,
        "sources_a": _format_sources(
            docs_a,
            paper_a,
        ),
        "sources_b": _format_sources(
            docs_b,
            paper_b,
        ),
    }
