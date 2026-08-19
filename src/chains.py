from langchain_core.output_parsers import StrOutputParser

from src.model_config import create_chat_model
from src.prompts import COMPARISON_PROMPT, QA_PROMPT
from src.vectorstore import retrieve


def get_llm():
    """Initialize the configured local or hosted chat model."""
    return create_chat_model()


def build_qa_chain():
    """Build a single-paper QA chain using LCEL."""
    llm = get_llm()
    output_parser = StrOutputParser()

    return QA_PROMPT.pipe(llm).pipe(output_parser)


def _format_context_with_metadata(docs: list) -> str:
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


def answer_question(
    question: str,
    paper_title: str | None,
) -> dict:
    """Answer a question using one paper or all indexed papers."""

    docs = retrieve(
        query=question,
        paper_filter=paper_title,
        k=4,
    )

    context_text = _format_context_with_metadata(docs)

    chain = build_qa_chain()

    answer = chain.invoke(
        {
            "context": context_text,
            "question": question,
        }
    )

    sources = [
        {
            "text": doc.page_content,
            "paper": doc.metadata.get(
                "paper_title",
                paper_title or "Unknown",
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

    return {
        "answer": answer,
        "sources": sources,
    }


def compare_papers(
    question: str,
    paper_a: str,
    paper_b: str,
) -> dict:
    """Compare two papers on a topic."""

    docs_a = retrieve(
        query=question,
        paper_filter=paper_a,
        k=3,
    )

    docs_b = retrieve(
        query=question,
        paper_filter=paper_b,
        k=3,
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

    def format_sources(
        docs: list,
        paper_title: str,
    ) -> list:
        return [
            {
                "text": doc.page_content,
                "paper": doc.metadata.get(
                    "paper_title",
                    paper_title,
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

    return {
        "comparison": comparison_table,
        "sources_a": format_sources(
            docs_a,
            paper_a,
        ),
        "sources_b": format_sources(
            docs_b,
            paper_b,
        ),
    }
