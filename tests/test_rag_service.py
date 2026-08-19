from types import SimpleNamespace

from paper_qa.application.rag_service import (
    QueryResult,
    RAGService,
)


def _make_service(
    mocker,
    *,
    load_pdf=None,
    chunkers=None,
    add_paper=None,
    delete_paper=None,
    list_papers=None,
    answer_question=None,
    compare_papers=None,
    classify_query=None,
):
    """Create a RAGService with controllable test dependencies."""

    return RAGService(
        load_pdf_fn=load_pdf or mocker.Mock(return_value=[]),
        chunkers=chunkers or {},
        add_paper_fn=add_paper or mocker.Mock(),
        delete_paper_fn=delete_paper or mocker.Mock(),
        list_papers_fn=list_papers or mocker.Mock(return_value=[]),
        answer_question_fn=answer_question or mocker.Mock(),
        compare_papers_fn=compare_papers or mocker.Mock(),
        classify_query_fn=classify_query or mocker.Mock(),
    )


def test_ingest_document_loads_chunks_and_stores_paper(mocker):
    pages = [
        SimpleNamespace(
            paper_title="Demo Paper",
        )
    ]

    chunks = [
        object(),
        object(),
    ]

    mock_loader = mocker.Mock(
        return_value=pages
    )

    mock_chunker = mocker.Mock(
        return_value=chunks
    )

    mock_add = mocker.Mock()
    mock_delete = mocker.Mock()

    service = _make_service(
        mocker,
        load_pdf=mock_loader,
        chunkers={
            "section_aware": mock_chunker,
        },
        add_paper=mock_add,
        delete_paper=mock_delete,
    )

    result = service.ingest_document(
        file_path="demo.pdf",
        chunking_strategy="section_aware",
    )

    mock_loader.assert_called_once_with(
        "demo.pdf"
    )

    mock_chunker.assert_called_once_with(
        pages
    )

    mock_delete.assert_called_once_with(
        "Demo Paper"
    )

    mock_add.assert_called_once_with(
        "Demo Paper",
        chunks,
    )

    assert result.paper_title == "Demo Paper"
    assert result.chunk_count == 2


def test_ingest_document_rejects_unknown_chunker(
    mocker,
):
    pages = [
        SimpleNamespace(
            paper_title="Demo Paper",
        )
    ]

    service = _make_service(
        mocker,
        load_pdf=mocker.Mock(
            return_value=pages
        ),
    )

    try:
        service.ingest_document(
            file_path="demo.pdf",
            chunking_strategy="unknown",
        )
    except ValueError as exc:
        assert "Unknown chunking strategy" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_run_query_single_selected_paper(
    mocker,
):
    mock_answer = mocker.Mock(
        return_value={
            "answer": "Answer for Paper A",
            "sources": [
                {
                    "paper": "Paper A",
                    "page": 1,
                    "text": "evidence",
                }
            ],
        }
    )

    service = _make_service(
        mocker,
        answer_question=mock_answer,
    )

    result = service.run_query(
        question="What is the result?",
        selected_papers=["Paper A"],
    )

    mock_answer.assert_called_once_with(
        "What is the result?",
        "Paper A",
    )

    assert isinstance(
        result,
        QueryResult,
    )

    assert result.answer == (
        "Answer for Paper A"
    )

    assert result.source_heading == (
        "Paper: Paper A"
    )


def test_run_query_two_selected_papers(
    mocker,
):
    mock_compare = mocker.Mock(
        return_value={
            "comparison": "Comparison result",
            "sources_a": [
                {
                    "paper": "Paper A",
                    "page": 1,
                    "text": "a",
                }
            ],
            "sources_b": [
                {
                    "paper": "Paper B",
                    "page": 2,
                    "text": "b",
                }
            ],
        }
    )

    service = _make_service(
        mocker,
        compare_papers=mock_compare,
    )

    result = service.run_query(
        question="Compare methodology",
        selected_papers=[
            "Paper A",
            "Paper B",
        ],
    )

    mock_compare.assert_called_once_with(
        "Compare methodology",
        "Paper A",
        "Paper B",
    )

    assert result.answer == (
        "Comparison result"
    )

    assert result.source_heading == (
        "Comparison: Paper A vs Paper B"
    )

    assert len(result.sources) == 2


def test_run_query_router_chooses_single_paper(
    mocker,
):
    mock_list = mocker.Mock(
        return_value=[
            "Paper A",
            "Paper B",
        ]
    )

    mock_router = mocker.Mock(
        return_value={
            "mode": "single_paper",
            "papers": ["Paper A"],
        }
    )

    mock_answer = mocker.Mock(
        return_value={
            "answer": "Single answer",
            "sources": [],
        }
    )

    service = _make_service(
        mocker,
        list_papers=mock_list,
        classify_query=mock_router,
        answer_question=mock_answer,
    )

    result = service.run_query(
        question="What does Paper A do?",
        selected_papers=[],
    )

    mock_router.assert_called_once_with(
        "What does Paper A do?",
        [
            "Paper A",
            "Paper B",
        ],
    )

    mock_answer.assert_called_once_with(
        "What does Paper A do?",
        "Paper A",
    )

    assert result.answer == "Single answer"
    assert result.source_heading == "Paper: Paper A"


def test_run_query_router_falls_back_to_all_papers(
    mocker,
):
    mock_list = mocker.Mock(
        return_value=[
            "Paper A",
            "Paper B",
        ]
    )

    mock_router = mocker.Mock(
        return_value={
            "mode": "single_paper",
            "papers": [],
        }
    )

    mock_answer = mocker.Mock(
        return_value={
            "answer": "Global answer",
            "sources": [],
        }
    )

    service = _make_service(
        mocker,
        list_papers=mock_list,
        classify_query=mock_router,
        answer_question=mock_answer,
    )

    result = service.run_query(
        question="What is deep learning?",
        selected_papers=None,
    )

    mock_answer.assert_called_once_with(
        "What is deep learning?",
        None,
    )

    assert result.answer == "Global answer"
    assert result.source_heading == "Paper: all papers"


def test_run_query_router_chooses_comparison(
    mocker,
):
    mock_list = mocker.Mock(
        return_value=[
            "Paper A",
            "Paper B",
        ]
    )

    mock_router = mocker.Mock(
        return_value={
            "mode": "comparison",
            "papers": [
                "Paper A",
                "Paper B",
            ],
        }
    )

    mock_compare = mocker.Mock(
        return_value={
            "comparison": "Routed comparison",
            "sources_a": [],
            "sources_b": [],
        }
    )

    service = _make_service(
        mocker,
        list_papers=mock_list,
        classify_query=mock_router,
        compare_papers=mock_compare,
    )

    result = service.run_query(
        question="Compare Paper A and Paper B",
        selected_papers=[],
    )

    mock_compare.assert_called_once_with(
        "Compare Paper A and Paper B",
        "Paper A",
        "Paper B",
    )

    assert result.answer == (
        "Routed comparison"
    )

    assert result.source_heading == (
        "Comparison: Paper A vs Paper B"
    )
