"""Tests for the retrieval policy layer."""

from langchain_core.documents import Document

from paper_qa.retrieval.service import (
    COMPARISON_RETRIEVAL_K,
    QUESTION_RETRIEVAL_K,
    retrieve_for_comparison,
    retrieve_for_question,
)


def test_retrieve_for_question_uses_question_policy(mocker):
    mock_retrieve = mocker.patch(
        "paper_qa.retrieval.service.retrieve",
        return_value=[
            Document(
                page_content="Relevant text.",
                metadata={
                    "paper_title": "Paper A",
                },
            )
        ],
    )

    result = retrieve_for_question(
        question="What method was used?",
        paper_title="Paper A",
    )

    mock_retrieve.assert_called_once_with(
        query="What method was used?",
        paper_filter="Paper A",
        k=QUESTION_RETRIEVAL_K,
    )

    assert len(result) == 1
    assert result[0].page_content == "Relevant text."


def test_retrieve_for_question_supports_all_papers(mocker):
    mock_retrieve = mocker.patch(
        "paper_qa.retrieval.service.retrieve",
        return_value=[],
    )

    retrieve_for_question(
        question="What approaches are discussed?",
        paper_title=None,
    )

    mock_retrieve.assert_called_once_with(
        query="What approaches are discussed?",
        paper_filter=None,
        k=QUESTION_RETRIEVAL_K,
    )


def test_retrieve_for_comparison_queries_each_paper(mocker):
    docs_a = [
        Document(
            page_content="Paper A evidence.",
            metadata={
                "paper_title": "Paper A",
            },
        )
    ]

    docs_b = [
        Document(
            page_content="Paper B evidence.",
            metadata={
                "paper_title": "Paper B",
            },
        )
    ]

    mock_retrieve = mocker.patch(
        "paper_qa.retrieval.service.retrieve"
    )

    mock_retrieve.side_effect = [
        docs_a,
        docs_b,
    ]

    result_a, result_b = retrieve_for_comparison(
        question="Compare methodology.",
        paper_a="Paper A",
        paper_b="Paper B",
    )

    assert mock_retrieve.call_count == 2

    mock_retrieve.assert_any_call(
        query="Compare methodology.",
        paper_filter="Paper A",
        k=COMPARISON_RETRIEVAL_K,
    )

    mock_retrieve.assert_any_call(
        query="Compare methodology.",
        paper_filter="Paper B",
        k=COMPARISON_RETRIEVAL_K,
    )

    assert result_a == docs_a
    assert result_b == docs_b
