"""Tests for src/chains.py."""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.chains import (
    _format_context_with_metadata,
    answer_question,
    compare_papers,
    get_llm,
)


def test_get_llm_initialization(mocker):
    """Ensure the configured chat model is initialized."""
    mock_create_chat_model = mocker.patch(
        "src.chains.create_chat_model"
    )

    get_llm()

    mock_create_chat_model.assert_called_once_with()


def test_format_context_with_metadata():
    """Context should contain page and section metadata."""
    docs = [
        Document(
            page_content="First chunk text.",
            metadata={
                "page_number": 1,
                "section": "Introduction",
            },
        ),
        Document(
            page_content="Second chunk text.",
            metadata={},
        ),
    ]

    formatted_text = _format_context_with_metadata(
        docs
    )

    assert (
        "[Page 1 | Section: Introduction]"
        in formatted_text
    )
    assert "First chunk text." in formatted_text

    assert (
        "[Page Unknown | Section: Unknown]"
        in formatted_text
    )
    assert "Second chunk text." in formatted_text
    assert "\n\n" in formatted_text


def test_answer_question_uses_retrieved_context_and_formats_output(
    mocker,
):
    """Q&A should consume retrieval results and generate an answer."""
    docs = [
        Document(
            page_content="Relevant text.",
            metadata={
                "paper_title": "Test Paper",
                "page_number": 5,
                "section": "2. Methods",
            },
        )
    ]

    mock_retrieve = mocker.patch(
        "src.chains.retrieve_for_question",
        return_value=docs,
    )

    mock_chain = mocker.Mock()
    mock_chain.invoke.return_value = (
        "This is a mocked LLM answer."
    )

    mocker.patch(
        "src.chains.build_qa_chain",
        return_value=mock_chain,
    )

    result = answer_question(
        "What method was used?",
        "Test Paper",
    )

    mock_retrieve.assert_called_once_with(
        question="What method was used?",
        paper_title="Test Paper",
    )

    mock_chain.invoke.assert_called_once()

    invocation = mock_chain.invoke.call_args.args[0]

    assert invocation["question"] == (
        "What method was used?"
    )
    assert "Relevant text." in invocation["context"]
    assert (
        "[Page 5 | Section: 2. Methods]"
        in invocation["context"]
    )

    assert result["answer"] == (
        "This is a mocked LLM answer."
    )

    assert result["sources"] == [
        {
            "text": "Relevant text.",
            "paper": "Test Paper",
            "page": 5,
            "section": "2. Methods",
        }
    ]


def test_answer_question_supports_all_papers(
    mocker,
):
    """None should represent an unfiltered all-paper question."""
    docs = [
        Document(
            page_content="Global evidence.",
            metadata={
                "paper_title": "Paper A",
                "page_number": 2,
                "section": "Results",
            },
        )
    ]

    mock_retrieve = mocker.patch(
        "src.chains.retrieve_for_question",
        return_value=docs,
    )

    mock_chain = mocker.Mock()
    mock_chain.invoke.return_value = "Answer"

    mocker.patch(
        "src.chains.build_qa_chain",
        return_value=mock_chain,
    )

    result = answer_question(
        "What do the papers conclude?",
        None,
    )

    mock_retrieve.assert_called_once_with(
        question="What do the papers conclude?",
        paper_title=None,
    )

    assert result["sources"][0]["paper"] == "Paper A"


def test_compare_papers_uses_retrieved_evidence_and_generates_comparison(
    mocker,
):
    """Comparison should consume evidence for both papers."""
    docs_a = [
        Document(
            page_content="A content.",
            metadata={
                "paper_title": "Paper A",
                "page_number": 10,
            },
        )
    ]

    docs_b = [
        Document(
            page_content="B content.",
            metadata={
                "paper_title": "Paper B",
                "page_number": 12,
            },
        )
    ]

    mock_retrieve = mocker.patch(
        "src.chains.retrieve_for_comparison",
        return_value=(docs_a, docs_b),
    )

    mock_llm_runnable = RunnableLambda(
        lambda _: (
            "| Aspect | Paper A | Paper B |\n"
            "|---|---|---|"
        )
    )

    mocker.patch(
        "src.chains.get_llm",
        return_value=mock_llm_runnable,
    )

    result = compare_papers(
        "Compare methodology",
        "Paper A",
        "Paper B",
    )

    mock_retrieve.assert_called_once_with(
        question="Compare methodology",
        paper_a="Paper A",
        paper_b="Paper B",
    )

    assert result["comparison"] == (
        "| Aspect | Paper A | Paper B |\n"
        "|---|---|---|"
    )

    assert len(result["sources_a"]) == 1
    assert (
        result["sources_a"][0]["text"]
        == "A content."
    )
    assert result["sources_a"][0]["page"] == 10
    assert (
        result["sources_a"][0]["paper"]
        == "Paper A"
    )

    assert len(result["sources_b"]) == 1
    assert (
        result["sources_b"][0]["text"]
        == "B content."
    )
    assert result["sources_b"][0]["page"] == 12
    assert (
        result["sources_b"][0]["paper"]
        == "Paper B"
    )
