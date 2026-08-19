"""Tests for the Gradio application handlers."""

from types import SimpleNamespace

import app


def test_runtime_status_shows_provider_without_machine_specific_model(mocker):
    mocker.patch(
        "app.get_app_model_name",
        return_value="ollama:qwen3-4b-local",
    )

    status = app.runtime_status(app.NAIVE_CHUNKING)

    assert "Answer provider:** Ollama" in status
    assert "qwen3-4b-local" not in status
    assert app.NAIVE_CHUNKING in status


def test_ingest_pdf_delegates_to_application_service(mocker):
    mock_ingest = mocker.patch(
        "app.ingest_document",
        return_value=SimpleNamespace(
            paper_title="Demo Paper",
            chunk_count=2,
        ),
    )

    mocker.patch(
        "app.list_papers",
        return_value=["Demo Paper"],
    )

    status, _, inventory = app.ingest_pdf(
        "demo.pdf",
        app.SECTION_AWARE_CHUNKING,
    )

    mock_ingest.assert_called_once_with(
        file_path="demo.pdf",
        chunking_strategy="section_aware",
    )

    assert app.SECTION_AWARE_CHUNKING in status
    assert "2 chunks" in status
    assert "Demo Paper" in inventory


def test_answer_query_uses_selected_single_paper(mocker):
    mocker.patch(
        "app.list_papers",
        return_value=["Paper A"],
    )

    mock_answer = mocker.patch(
        "app.answer_question",
        return_value={
            "answer": "A cited answer [p. 2].",
            "sources": [
                {
                    "paper": "Paper A",
                    "page": 2,
                    "section": "Methods",
                    "text": "Evidence",
                }
            ],
        },
    )

    cleared, history, sources = app.answer_query(
        "What method was used?",
        [],
        ["Paper A"],
    )

    mock_answer.assert_called_once_with(
        "What method was used?",
        "Paper A",
    )

    assert cleared == ""
    assert history[-1]["content"] == "A cited answer [p. 2]."
    assert "Paper A — page 2 — Methods" in sources


def test_format_sources_omits_unknown_or_empty_sections():
    sources = [
        {
            "paper": "Paper A",
            "page": 1,
            "section": "Unknown",
            "text": "A",
        },
        {
            "paper": "Paper B",
            "page": 2,
            "section": None,
            "text": "B",
        },
        {
            "paper": "Paper C",
            "page": 3,
            "section": "",
            "text": "C",
        },
    ]

    formatted = app._format_sources(sources)

    assert "Paper A — page 1**" in formatted
    assert "Paper B — page 2**" in formatted
    assert "Paper C — page 3**" in formatted
    assert "— Unknown" not in formatted


def test_answer_query_uses_two_selected_papers_for_comparison(mocker):
    mocker.patch(
        "app.list_papers",
        return_value=["Paper A", "Paper B"],
    )

    mock_compare = mocker.patch(
        "app.compare_papers",
        return_value={
            "comparison": "| Aspect | Paper A | Paper B |",
            "sources_a": [],
            "sources_b": [],
        },
    )

    _, history, sources = app.answer_query(
        "Compare the methods.",
        [],
        ["Paper A", "Paper B"],
    )

    mock_compare.assert_called_once_with(
        "Compare the methods.",
        "Paper A",
        "Paper B",
    )

    assert history[-1]["content"] == "| Aspect | Paper A | Paper B |"
    assert sources == "_No sources were returned._"
