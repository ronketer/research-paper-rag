"""Tests for src/vectorstore.py (using pytest-mock)"""

import pytest
from langchain_core.documents import Document

from src.chunker import Chunk
from src.vectorstore import (
    add_paper,
    delete_paper,
    list_papers,
    retrieve,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(paper_title: str, count: int = 3) -> list[Chunk]:
    """Create a list of Chunk objects for testing."""
    return [
        Chunk(
            text=f"Sample text for chunk {i}",
            paper_title=paper_title,
            section="1. Introduction",
            page_number=1,
            chunk_index=i
        )
        for i in range(count)
    ]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_add_paper_creates_correct_documents(mocker):
    """add_paper should convert Chunks to LangChain Documents with exact metadata."""
    mock_db = mocker.Mock()
    mocker.patch("src.vectorstore.get_vectorstore", return_value=mock_db)

    chunks = _make_chunks("Test Paper 1", count=2)
    add_paper("Test Paper 1", chunks)

    mock_db.add_documents.assert_called_once()

    added_docs = mock_db.add_documents.call_args[0][0]

    assert len(added_docs) == 2
    assert isinstance(added_docs[0], Document)
    assert added_docs[0].page_content == "Sample text for chunk 0"

    expected_metadata = {
        "paper_title": "Test Paper 1",
        "page_number": 1,
        "chunk_index": 0,
        "section": "1. Introduction"
    }
    assert added_docs[0].metadata == expected_metadata


def test_retrieve_no_filter(mocker):
    """If no paper_filter is provided, it should search globally."""
    mock_db = mocker.Mock()
    mocker.patch("src.vectorstore.get_vectorstore", return_value=mock_db)

    retrieve("What is attention?", k=5)

    mock_db.similarity_search.assert_called_once_with("What is attention?", k=5)


def test_retrieve_single_string_filter(mocker):
    """If paper_filter is a string, it should pass a single exact-match filter."""
    mock_db = mocker.Mock()
    mocker.patch("src.vectorstore.get_vectorstore", return_value=mock_db)

    retrieve("What is attention?", paper_filter="Attention Is All You Need", k=4)

    expected_filter = {"paper_title": "Attention Is All You Need"}
    mock_db.similarity_search.assert_called_once_with(
        "What is attention?",
        k=4,
        filter=expected_filter
    )


def test_retrieve_list_filter_distributes_k(mocker):
    """If paper_filter is a list, it should split k evenly across the papers."""
    mock_db = mocker.Mock()
    mock_doc = Document(page_content="test result")
    mock_db.similarity_search.return_value = [mock_doc, mock_doc]
    mocker.patch("src.vectorstore.get_vectorstore", return_value=mock_db)

    results = retrieve("Compare results", paper_filter=["Paper A", "Paper B"], k=5)

    assert mock_db.similarity_search.call_count == 2
    mock_db.similarity_search.assert_any_call(
        "Compare results",
        k=3,
        filter={"paper_title": "Paper A"}
    )
    mock_db.similarity_search.assert_any_call(
        "Compare results",
        k=3,
        filter={"paper_title": "Paper B"}
    )
    assert len(results) <= 5


def test_list_papers_returns_unique_sorted_titles(mocker):
    """list_papers should extract unique titles from all chunk metadata and sort them."""
    mock_db = mocker.Mock()
    mock_db.get.return_value = {
        "metadatas": [
            {"paper_title": "Zebra Dynamics"},
            {"paper_title": "Apple Biology"},
            {"paper_title": "Zebra Dynamics"},
            {}
        ]
    }
    mocker.patch("src.vectorstore.get_vectorstore", return_value=mock_db)

    titles = list_papers()

    assert titles == ["Apple Biology", "Zebra Dynamics"]
    mock_db.get.assert_called_once_with(include=["metadatas"])


def test_delete_paper(mocker):
    """delete_paper should trigger the Chroma delete method with the correct where clause."""
    mock_db = mocker.Mock()
    mocker.patch("src.vectorstore.get_vectorstore", return_value=mock_db)

    delete_paper("Bad Paper")

    mock_db.delete.assert_called_once_with(where={"paper_title": "Bad Paper"})
