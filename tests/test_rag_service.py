from types import SimpleNamespace

from src.application import rag_service
from src.application.rag_service import QueryResult, ingest_document, run_query


def test_ingest_document_loads_chunks_and_stores_paper(mocker):
    pages = [
        SimpleNamespace(paper_title="Demo Paper"),
    ]
    chunks = [object(), object()]

    mocker.patch(
        "src.application.rag_service.load_pdf",
        return_value=pages,
    )

    selected_chunker = mocker.Mock(return_value=chunks)
    mocker.patch.dict(
        rag_service.CHUNKERS,
        {"section_aware": selected_chunker},
    )

    mock_delete = mocker.patch(
        "src.application.rag_service.delete_paper"
    )
    mock_add = mocker.patch(
        "src.application.rag_service.add_paper"
    )

    result = ingest_document(
        file_path="demo.pdf",
        chunking_strategy="section_aware",
    )

    selected_chunker.assert_called_once_with(pages)
    mock_delete.assert_called_once_with("Demo Paper")
    mock_add.assert_called_once_with("Demo Paper", chunks)

    assert result.paper_title == "Demo Paper"
    assert result.chunk_count == 2


def test_run_query_single_selected_paper(mocker):
    mock_answer = mocker.patch(
        "src.application.rag_service.answer_question",
        return_value={
            "answer": "Answer for Paper A",
            "sources": [{"paper": "Paper A", "page": 1, "text": "evidence"}],
        },
    )

    result = run_query(
        question="What is the result?",
        selected_papers=["Paper A"],
    )

    mock_answer.assert_called_once_with("What is the result?", "Paper A")
    assert isinstance(result, QueryResult)
    assert result.answer == "Answer for Paper A"
    assert result.sources == [{"paper": "Paper A", "page": 1, "text": "evidence"}]
    assert result.source_heading == "Paper: Paper A"


def test_run_query_two_selected_papers(mocker):
    mock_compare = mocker.patch(
        "src.application.rag_service.compare_papers",
        return_value={
            "comparison": "Comparison table between Paper A and Paper B",
            "sources_a": [{"paper": "Paper A", "page": 1, "text": "evidence a"}],
            "sources_b": [{"paper": "Paper B", "page": 2, "text": "evidence b"}],
        },
    )

    result = run_query(
        question="Compare methodology",
        selected_papers=["Paper A", "Paper B"],
    )

    mock_compare.assert_called_once_with(
        "Compare methodology", "Paper A", "Paper B"
    )
    assert isinstance(result, QueryResult)
    assert result.answer == "Comparison table between Paper A and Paper B"
    assert result.sources == [
        {"paper": "Paper A", "page": 1, "text": "evidence a"},
        {"paper": "Paper B", "page": 2, "text": "evidence b"},
    ]
    assert result.source_heading == "Comparison: Paper A vs Paper B"


def test_run_query_no_selection_router_chooses_single_paper(mocker):
    mocker.patch(
        "src.application.rag_service.list_papers",
        return_value=["Paper A", "Paper B"],
    )
    mock_classify = mocker.patch(
        "src.application.rag_service.classify_query",
        return_value={
            "mode": "single_paper",
            "papers": ["Paper A"],
        },
    )
    mock_answer = mocker.patch(
        "src.application.rag_service.answer_question",
        return_value={
            "answer": "Single paper answer",
            "sources": [{"paper": "Paper A", "page": 3, "text": "context"}],
        },
    )

    result = run_query(
        question="What does Paper A do?",
        selected_papers=[],
    )

    mock_classify.assert_called_once_with(
        "What does Paper A do?", ["Paper A", "Paper B"]
    )
    mock_answer.assert_called_once_with("What does Paper A do?", "Paper A")
    assert isinstance(result, QueryResult)
    assert result.answer == "Single paper answer"
    assert result.sources == [{"paper": "Paper A", "page": 3, "text": "context"}]
    assert result.source_heading == "Paper: Paper A"


def test_run_query_no_selection_router_chooses_fallback_all_papers(mocker):
    mocker.patch(
        "src.application.rag_service.list_papers",
        return_value=["Paper A", "Paper B"],
    )
    mock_classify = mocker.patch(
        "src.application.rag_service.classify_query",
        return_value={
            "mode": "single_paper",
            "papers": [],
        },
    )
    mock_answer = mocker.patch(
        "src.application.rag_service.answer_question",
        return_value={
            "answer": "Global answer across papers",
            "sources": [{"paper": "Paper B", "page": 1, "text": "context"}],
        },
    )

    result = run_query(
        question="What is deep learning?",
        selected_papers=None,
    )

    mock_classify.assert_called_once_with(
        "What is deep learning?", ["Paper A", "Paper B"]
    )
    mock_answer.assert_called_once_with("What is deep learning?", None)
    assert isinstance(result, QueryResult)
    assert result.answer == "Global answer across papers"
    assert result.sources == [{"paper": "Paper B", "page": 1, "text": "context"}]
    assert result.source_heading == "Paper: all papers"


def test_run_query_no_selection_router_chooses_comparison(mocker):
    mocker.patch(
        "src.application.rag_service.list_papers",
        return_value=["Paper A", "Paper B"],
    )
    mock_classify = mocker.patch(
        "src.application.rag_service.classify_query",
        return_value={
            "mode": "comparison",
            "papers": ["Paper A", "Paper B"],
        },
    )
    mock_compare = mocker.patch(
        "src.application.rag_service.compare_papers",
        return_value={
            "comparison": "Routed comparison result",
            "sources_a": [{"paper": "Paper A", "page": 5, "text": "a"}],
            "sources_b": [{"paper": "Paper B", "page": 7, "text": "b"}],
        },
    )

    result = run_query(
        question="Compare Paper A and Paper B",
        selected_papers=[],
    )

    mock_classify.assert_called_once_with(
        "Compare Paper A and Paper B", ["Paper A", "Paper B"]
    )
    mock_compare.assert_called_once_with(
        "Compare Paper A and Paper B", "Paper A", "Paper B"
    )
    assert isinstance(result, QueryResult)
    assert result.answer == "Routed comparison result"
    assert result.sources == [
        {"paper": "Paper A", "page": 5, "text": "a"},
        {"paper": "Paper B", "page": 7, "text": "b"},
    ]
    assert result.source_heading == "Comparison: Paper A vs Paper B"

