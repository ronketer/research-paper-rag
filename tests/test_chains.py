"""Tests for src/chains.py (using pytest-mock)"""

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.chains import (
    _format_context_with_metadata,
    answer_question,
    compare_papers,
    get_llm,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_llm_initialization(mocker):
    """Ensure the LLM is initialized with the correct model and temperature."""
    mock_create_chat_model = mocker.patch("src.chains.create_chat_model")
    get_llm()

    mock_create_chat_model.assert_called_once_with()


def test_format_context_with_metadata():
    """Ensure context chunks are formatted cleanly with page and section info."""
    docs = [
        Document(
            page_content="First chunk text.",
            metadata={"page_number": 1, "section": "Introduction"}
        ),
        Document(
            page_content="Second chunk text.",
            metadata={}
        )
    ]

    formatted_text = _format_context_with_metadata(docs)

    assert "[Page 1 | Section: Introduction]" in formatted_text
    assert "First chunk text." in formatted_text

    assert "[Page Unknown | Section: Unknown]" in formatted_text
    assert "Second chunk text." in formatted_text
    assert "\n\n" in formatted_text


def test_answer_question_formats_output_correctly(mocker):
    """Test that a single-paper QA query fetches docs, triggers the chain, and builds the source list."""

    # Mock Retriever
    mock_retrieve = mocker.patch("src.chains.retrieve")
    mock_retrieve.return_value = [
        Document(page_content="Relevant text.", metadata={"page_number": 5, "section": "2. Methods"})
    ]

    # Mock Chain
    mock_chain = mocker.Mock()
    mock_chain.invoke.return_value = "This is a mocked LLM answer."
    mocker.patch("src.chains.build_qa_chain", return_value=mock_chain)

    # Execute
    result = answer_question("What method was used?", "Test Paper")

    # Assertions
    mock_retrieve.assert_called_once_with(
        query="What method was used?",
        paper_filter="Test Paper",
        k=4
    )

    assert result["answer"] == "This is a mocked LLM answer."
    assert len(result["sources"]) == 1
    assert result["sources"][0] == {
        "text": "Relevant text.",
        "paper": "Test Paper",
        "page": 5,
        "section": "2. Methods"
    }


def test_compare_papers_structures_multi_document_comparison(mocker):
    """Test that comparison retrieves for two distinct papers and runs the LCEL prompt."""

    # Mock Retriever
    mock_retrieve = mocker.patch("src.chains.retrieve")
    def fake_retrieve(query, paper_filter, k):
        if paper_filter == "Paper A":
            return [Document(page_content="A content.", metadata={"page_number": 10})]
        if paper_filter == "Paper B":
            return [Document(page_content="B content.", metadata={"page_number": 12})]
        return []
    mock_retrieve.side_effect = fake_retrieve

    # Mock LLM component in the pipeline
    mock_llm_runnable = RunnableLambda(lambda x: "| Aspect | Paper A | Paper B |\n|---|---|---|")
    mocker.patch("src.chains.get_llm", return_value=mock_llm_runnable)

    # Execute
    result = compare_papers("Compare methodology", "Paper A", "Paper B")

    # Assertions
    assert mock_retrieve.call_count == 2
    mock_retrieve.assert_any_call(query="Compare methodology", paper_filter="Paper A", k=3)
    mock_retrieve.assert_any_call(query="Compare methodology", paper_filter="Paper B", k=3)

    assert result["comparison"] == "| Aspect | Paper A | Paper B |\n|---|---|---|"

    assert len(result["sources_a"]) == 1
    assert result["sources_a"][0]["text"] == "A content."
    assert result["sources_a"][0]["page"] == 10

    assert len(result["sources_b"]) == 1
    assert result["sources_b"][0]["text"] == "B content."
    assert result["sources_b"][0]["page"] == 12
