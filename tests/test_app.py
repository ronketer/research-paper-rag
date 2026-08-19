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


def test_answer_query_delegates_to_run_query_and_updates_history(mocker):
    mocker.patch(
        "app.list_papers",
        return_value=["Paper A"],
    )

    mock_run = mocker.patch(
        "app.run_query",
        return_value=SimpleNamespace(
            answer="A cited answer [p. 2].",
            sources=[
                {
                    "paper": "Paper A",
                    "page": 2,
                    "section": "Methods",
                    "text": "Evidence",
                }
            ],
            source_heading="Paper: Paper A",
        ),
    )

    cleared, history, sources = app.answer_query(
        "What method was used?",
        [],
        ["Paper A"],
    )

    mock_run.assert_called_once_with(
        question="What method was used?",
        selected_papers=["Paper A"],
    )

    assert cleared == ""
    assert history[-1]["content"] == "A cited answer [p. 2]."
    assert "Paper A — page 2 — Methods" in sources
    assert "### Paper: Paper A" in sources


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


def test_answer_query_delegates_two_selected_papers_and_formats_empty_sources(mocker):
    mocker.patch(
        "app.list_papers",
        return_value=["Paper A", "Paper B"],
    )

    mock_run = mocker.patch(
        "app.run_query",
        return_value=SimpleNamespace(
            answer="| Aspect | Paper A | Paper B |",
            sources=[],
            source_heading="Comparison: Paper A vs Paper B",
        ),
    )

    _, history, sources = app.answer_query(
        "Compare the methods.",
        [],
        ["Paper A", "Paper B"],
    )

    mock_run.assert_called_once_with(
        question="Compare the methods.",
        selected_papers=["Paper A", "Paper B"],
    )

    assert history[-1]["content"] == "| Aspect | Paper A | Paper B |"
    assert sources == "_No sources were returned._"
