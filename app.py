"""Small Gradio demo for the research-paper Q&A pipeline."""

from __future__ import annotations

import argparse
import os
from typing import Any

import gradio as gr

from paper_qa.bootstrap import (
    delete_ingested_paper,
    ingest_document,
    list_ingested_papers,
    run_query,
)
from paper_qa.config.model_config import get_app_model_name

ChatHistory = list[dict[str, Any]]

NAIVE_CHUNKING = "Naive — benchmark winner"
SECTION_AWARE_CHUNKING = "Section-aware — experimental"
DEFAULT_CHUNKING_STRATEGY = NAIVE_CHUNKING

CHUNKING_STRATEGIES = {
    NAIVE_CHUNKING: "naive",
    SECTION_AWARE_CHUNKING: "section_aware",
}


def runtime_status(
    chunking_strategy: str,
) -> str:
    """Describe the currently configured model and chunker."""

    model_name = get_app_model_name()

    provider_name = (
        model_name
        .split(":", maxsplit=1)[0]
        .replace("-", " ")
        .title()
        if ":" in model_name
        else "Custom"
    )

    return (
        f"**Answer provider:** {provider_name} "
        f"(configured in `.env`)  \n"
        f"**Ingestion chunker:** {chunking_strategy}"
    )


def _paper_inventory(
    papers: list[str],
) -> str:
    """Format stored paper names for the UI."""

    if not papers:
        return "_No papers have been ingested yet._"

    return "\n".join(
        f"- {paper}"
        for paper in papers
    )


def _dropdown_update(
    papers: list[str],
    selected: list[str] | None = None,
) -> gr.Dropdown:
    """Build an updated Gradio paper selector."""

    selected = [
        paper
        for paper in (selected or [])
        if paper in papers
    ]

    return gr.Dropdown(
        choices=papers,
        value=selected,
        multiselect=True,
        max_choices=2,
    )


def refresh_papers() -> tuple[gr.Dropdown, str]:
    """Refresh papers when the UI opens."""

    papers = list_ingested_papers()

    return (
        _dropdown_update(papers),
        _paper_inventory(papers),
    )


def ingest_pdf(
    file_path: str | None,
    chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY,
) -> tuple[str, gr.Dropdown, str]:
    """Ingest a PDF selected in the Gradio uploader."""

    if not file_path:
        papers = list_ingested_papers()

        return (
            "Please choose a PDF first.",
            _dropdown_update(papers),
            _paper_inventory(papers),
        )

    try:
        strategy = CHUNKING_STRATEGIES.get(
            chunking_strategy
        )

        if strategy is None:
            raise ValueError(
                "Unknown chunking strategy: "
                f"{chunking_strategy}"
            )

        result = ingest_document(
            file_path=file_path,
            chunking_strategy=strategy,
        )

        papers = list_ingested_papers()

        status = (
            f"✅ Ingested **{result.paper_title}** "
            f"using **{chunking_strategy}** "
            f"({result.chunk_count} chunks)."
        )

        return (
            status,
            _dropdown_update(
                papers,
                [result.paper_title],
            ),
            _paper_inventory(papers),
        )

    except Exception as exc:
        papers = list_ingested_papers()

        return (
            f"❌ Ingestion failed: {exc}",
            _dropdown_update(papers),
            _paper_inventory(papers),
        )


def remove_selected_papers(
    selected_papers: list[str] | None,
) -> tuple[str, gr.Dropdown, str]:
    """Delete selected papers."""

    selected_papers = (
        selected_papers or []
    )

    if not selected_papers:
        papers = list_ingested_papers()

        return (
            "Select a paper to delete.",
            _dropdown_update(papers),
            _paper_inventory(papers),
        )

    for paper in selected_papers:
        delete_ingested_paper(paper)

    papers = list_ingested_papers()
    removed = ", ".join(selected_papers)

    return (
        f"🗑️ Deleted: **{removed}**",
        _dropdown_update(papers),
        _paper_inventory(papers),
    )


def _format_sources(
    sources: list[dict[str, Any]],
    heading: str = "Retrieved sources",
) -> str:
    """Format retrieved evidence as Markdown."""

    if not sources:
        return "_No sources were returned._"

    sections = [
        f"### {heading}"
    ]

    for source in sources:
        paper = source.get(
            "paper",
            "Unknown paper",
        )
        page = source.get(
            "page",
            "Unknown",
        )
        section = source.get("section")

        section_text = (
            str(section).strip()
            if section is not None
            else ""
        )

        source_label = (
            f"{paper} — page {page}"
        )

        if (
            section_text
            and section_text.casefold()
            != "unknown"
        ):
            source_label += (
                f" — {section_text}"
            )

        quoted_text = "\n".join(
            f"> {line}" if line else ">"
            for line in str(
                source.get("text", "")
            ).splitlines()
        )

        sections.append(
            f"**{source_label}**\n\n"
            f"{quoted_text}"
        )

    return "\n\n---\n\n".join(
        sections
    )


def answer_query(
    question: str,
    history: ChatHistory | None,
    selected_papers: list[str] | None,
) -> tuple[str, ChatHistory, str]:
    """Run Q&A or comparison and update Gradio state."""

    question = (
        question or ""
    ).strip()

    history = list(
        history or []
    )

    selected_papers = (
        selected_papers or []
    )

    if not question:
        return (
            "",
            history,
            "_Type a question first._",
        )

    history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    available_papers = (
        list_ingested_papers()
    )

    if not available_papers:
        history.append(
            {
                "role": "assistant",
                "content": (
                    "Upload and ingest at least one PDF "
                    "before asking a question."
                ),
            }
        )

        return (
            "",
            history,
            "_No sources available._",
        )

    try:
        result = run_query(
            question=question,
            selected_papers=selected_papers,
        )

        history.append(
            {
                "role": "assistant",
                "content": result.answer,
            }
        )

        return (
            "",
            history,
            _format_sources(
                result.sources,
                result.source_heading,
            ),
        )

    except Exception as exc:
        configured_model = (
            get_app_model_name()
        )

        if configured_model.startswith(
            "ollama:"
        ):
            help_text = (
                "Ensure Ollama is running and the "
                "configured model is installed: "
                f"{configured_model}."
            )
        else:
            help_text = (
                "Verify the provider integration, "
                "API credentials, and configured "
                f"model name: {configured_model}."
            )

        history.append(
            {
                "role": "assistant",
                "content": (
                    "Could not generate an answer: "
                    f"{exc}\n\n{help_text}"
                ),
            }
        )

        return (
            "",
            history,
            "_No sources available because "
            "the request failed._",
        )


with gr.Blocks(
    title="Research Paper Q&A",
    delete_cache=(86_400, 86_400),
) as demo:
    gr.Markdown(
        """
        # Research Paper Q&A
        Upload academic PDFs, ask cited questions,
        or select two papers to compare.
        """
    )

    with gr.Row():
        with gr.Column(
            scale=1,
            min_width=300,
        ):
            chunking_selector = gr.Radio(
                choices=list(
                    CHUNKING_STRATEGIES
                ),
                value=DEFAULT_CHUNKING_STRATEGY,
                label="Chunking strategy",
                info=(
                    "Naive is the current benchmark "
                    "winner. Section-aware is available "
                    "as an experimental comparison."
                ),
            )

            runtime_markdown = gr.Markdown(
                runtime_status(
                    DEFAULT_CHUNKING_STRATEGY
                )
            )

            pdf_file = gr.File(
                label="Upload a PDF",
                file_types=[".pdf"],
                type="filepath",
            )

            ingest_button = gr.Button(
                "Process and ingest",
                variant="primary",
            )

            ingestion_status = gr.Markdown()

            paper_selector = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                max_choices=2,
                label="Papers for the next question",
                info=(
                    "Choose one for Q&A, two for "
                    "comparison, or none for automatic "
                    "routing."
                ),
            )

            delete_button = gr.Button(
                "Delete selected from database",
                variant="stop",
            )

            gr.Markdown(
                "### Ingested papers"
            )

            paper_list = gr.Markdown(
                "_Loading papers..._"
            )

        with gr.Column(
            scale=3,
            min_width=500,
        ):
            chatbot = gr.Chatbot(
                label="Q&A session",
                height=480,
                placeholder=(
                    "Ingest a paper, then ask a question."
                ),
            )

            gr.Markdown(
                "_Questions are currently independent; "
                "earlier messages are displayed but are "
                "not passed back to the model._"
            )

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder=(
                        "Ask a question about the "
                        "selected paper(s)..."
                    ),
                    show_label=False,
                    scale=5,
                )

                ask_button = gr.Button(
                    "Ask",
                    variant="primary",
                    scale=1,
                )

            gr.Examples(
                examples=[
                    [
                        "Which generator model does "
                        "the RAG paper use?"
                    ],
                    [
                        "Why are in-batch negatives "
                        "efficient for training DPR?"
                    ],
                    [
                        "Compare the passage collections "
                        "used by DPR and RAG."
                    ],
                ],
                inputs=question_box,
                label=(
                    "Example questions for the bundled papers"
                ),
            )

            with gr.Accordion(
                "Retrieved sources",
                open=False,
            ):
                sources_markdown = gr.Markdown(
                    "_Sources appear after an answer._"
                )

            gr.ClearButton(
                [
                    question_box,
                    chatbot,
                    sources_markdown,
                ],
                value="Clear Q&A session",
            )

    demo.load(
        refresh_papers,
        outputs=[
            paper_selector,
            paper_list,
        ],
    )

    ingest_button.click(
        ingest_pdf,
        inputs=[
            pdf_file,
            chunking_selector,
        ],
        outputs=[
            ingestion_status,
            paper_selector,
            paper_list,
        ],
    )

    chunking_selector.change(
        runtime_status,
        inputs=chunking_selector,
        outputs=runtime_markdown,
    )

    delete_button.click(
        remove_selected_papers,
        inputs=paper_selector,
        outputs=[
            ingestion_status,
            paper_selector,
            paper_list,
        ],
    )

    ask_button.click(
        answer_query,
        inputs=[
            question_box,
            chatbot,
            paper_selector,
        ],
        outputs=[
            question_box,
            chatbot,
            sources_markdown,
        ],
    )

    question_box.submit(
        answer_query,
        inputs=[
            question_box,
            chatbot,
            paper_selector,
        ],
        outputs=[
            question_box,
            chatbot,
            sources_markdown,
        ],
    )


def main() -> None:
    """Launch the Gradio application."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help=(
            "Create a temporary public Gradio link."
        ),
    )

    args = parser.parse_args()

    share_demo = (
        args.share
        or os.getenv(
            "GRADIO_SHARE",
            "false",
        ).lower()
        == "true"
    )

    demo.queue(
        default_concurrency_limit=1
    ).launch(
        share=share_demo,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
