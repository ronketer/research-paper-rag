"""Small Gradio demo for the research-paper Q&A pipeline."""

from __future__ import annotations

import argparse
import os
from typing import Any

import gradio as gr

from src.chains import answer_question, compare_papers
from src.chunker import naive_chunk, section_aware_chunk
from src.loader import load_pdf
from src.model_config import get_app_model_name
from src.router import classify_query
from src.vectorstore import add_paper, delete_paper, list_papers


# Gradio's Chatbot uses a list of dictionaries in this shape.
ChatHistory = list[dict[str, Any]]

NAIVE_CHUNKING = "Naive — benchmark winner"
SECTION_AWARE_CHUNKING = "Section-aware — experimental"
DEFAULT_CHUNKING_STRATEGY = NAIVE_CHUNKING
CHUNKING_STRATEGIES = {
    NAIVE_CHUNKING: naive_chunk,
    SECTION_AWARE_CHUNKING: section_aware_chunk,
}


def runtime_status(chunking_strategy: str) -> str:
    """Describe the model and ingestion strategy currently selected in the UI."""
    model_name = get_app_model_name()
    provider_name = (
        model_name.split(":", maxsplit=1)[0].replace("-", " ").title()
        if ":" in model_name
        else "Custom"
    )
    return (
        f"**Answer provider:** {provider_name} (configured in `.env`)  \n"
        f"**Ingestion chunker:** {chunking_strategy}"
    )


def _paper_inventory(papers: list[str]) -> str:
    """Turn the stored paper names into a short Markdown list."""
    if not papers:
        return "_No papers have been ingested yet._"
    return "\n".join(f"- {paper}" for paper in papers)


def _dropdown_update(
    papers: list[str],
    selected: list[str] | None = None,
) -> gr.Dropdown:
    """Return a new Dropdown configuration for Gradio to apply in the UI."""
    selected = [paper for paper in (selected or []) if paper in papers]
    return gr.Dropdown(
        choices=papers,
        value=selected,
        multiselect=True,
        max_choices=2,
    )


def refresh_papers() -> tuple[gr.Dropdown, str]:
    """Refresh the paper selector when the demo first opens."""
    papers = list_papers()
    return _dropdown_update(papers), _paper_inventory(papers)


def ingest_pdf(
    file_path: str | None,
    chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY,
) -> tuple[str, gr.Dropdown, str]:
    """Extract, chunk, and store one PDF selected in the Gradio uploader."""
    if not file_path:
        papers = list_papers()
        return "Please choose a PDF first.", _dropdown_update(papers), _paper_inventory(papers)

    try:
        # Gradio gives us a temporary filepath, which PyMuPDF can read directly.
        pages = load_pdf(file_path)
        if not pages:
            raise ValueError("No text could be extracted from this PDF.")

        chunker = CHUNKING_STRATEGIES.get(chunking_strategy)
        if chunker is None:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

        title = pages[0].paper_title
        chunks = chunker(pages)

        # Replace an older copy instead of adding duplicate chunks.
        delete_paper(title)
        add_paper(title, chunks)

        papers = list_papers()
        status = (
            f"✅ Ingested **{title}** using **{chunking_strategy}** "
            f"({len(chunks)} chunks)."
        )
        return status, _dropdown_update(papers, [title]), _paper_inventory(papers)
    except Exception as exc:
        papers = list_papers()
        status = f"❌ Ingestion failed: {exc}"
        return status, _dropdown_update(papers), _paper_inventory(papers)


def remove_selected_papers(
    selected_papers: list[str] | None,
) -> tuple[str, gr.Dropdown, str]:
    """Delete the papers currently selected in the dropdown."""
    selected_papers = selected_papers or []
    if not selected_papers:
        papers = list_papers()
        return "Select a paper to delete.", _dropdown_update(papers), _paper_inventory(papers)

    for paper in selected_papers:
        delete_paper(paper)

    papers = list_papers()
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
    """Format retrieved chunks as readable Markdown below the chat."""
    if not sources:
        return "_No sources were returned._"

    sections = [f"### {heading}"]
    for source in sources:
        paper = source.get("paper", "Unknown paper")
        page = source.get("page", "Unknown")
        section = source.get("section")
        section_text = str(section).strip() if section is not None else ""
        source_label = f"{paper} — page {page}"
        if section_text and section_text.casefold() != "unknown":
            source_label += f" — {section_text}"

        # Prefix each source line with > so PDF text is displayed as a quote.
        quoted_text = "\n".join(
            f"> {line}" if line else ">"
            for line in str(source.get("text", "")).splitlines()
        )
        sections.append(
            f"**{source_label}**\n\n{quoted_text}"
        )
    return "\n\n---\n\n".join(sections)


def answer_query(
    question: str,
    history: ChatHistory | None,
    selected_papers: list[str] | None,
) -> tuple[str, ChatHistory, str]:
    """Run Q&A or comparison mode and return updated Gradio components."""
    question = (question or "").strip()
    history = list(history or [])
    selected_papers = selected_papers or []

    if not question:
        return "", history, "_Type a question first._"

    # Add the user's message before doing slower retrieval/model work.
    history.append({"role": "user", "content": question})
    available_papers = list_papers()
    if not available_papers:
        history.append(
            {
                "role": "assistant",
                "content": "Upload and ingest at least one PDF before asking a question.",
            }
        )
        return "", history, "_No sources available._"

    try:
        # Explicit UI selection takes priority over automatic query routing.
        if len(selected_papers) == 2:
            paper_a, paper_b = selected_papers
            result = compare_papers(question, paper_a, paper_b)
            answer = result["comparison"]
            sources = result["sources_a"] + result["sources_b"]
            source_heading = f"Comparison: {paper_a} vs {paper_b}"
        elif len(selected_papers) == 1:
            paper = selected_papers[0]
            result = answer_question(question, paper)
            answer = result["answer"]
            sources = result["sources"]
            source_heading = f"Paper: {paper}"
        else:
            # With no manual selection, preserve the project's existing router.
            route = classify_query(question, available_papers)
            if route["mode"] == "comparison" and len(route["papers"]) >= 2:
                paper_a, paper_b = route["papers"][:2]
                result = compare_papers(question, paper_a, paper_b)
                answer = result["comparison"]
                sources = result["sources_a"] + result["sources_b"]
                source_heading = f"Comparison: {paper_a} vs {paper_b}"
            else:
                paper = route["papers"][0] if route["papers"] else None
                result = answer_question(question, paper)
                answer = result["answer"]
                sources = result["sources"]
                source_heading = f"Paper: {paper or 'all papers'}"

        history.append({"role": "assistant", "content": answer})
        return "", history, _format_sources(sources, source_heading)
    except Exception as exc:
        configured_model = get_app_model_name()
        if configured_model.startswith("ollama:"):
            help_text = (
                "Ensure Ollama is running and the configured model is installed: "
                f"{configured_model}."
            )
        else:
            help_text = (
                "Verify the provider integration, API credentials, and configured "
                f"model name: {configured_model}."
            )

        history.append(
            {
                "role": "assistant",
                "content": f"Could not generate an answer: {exc}\n\n{help_text}",
            }
        )
        return "", history, "_No sources available because the request failed._"


# Blocks is used instead of Interface because this demo has several connected
# actions: PDF ingestion, paper selection, chat, deletion, and source display.
with gr.Blocks(
    title="Research Paper Q&A",
    delete_cache=(86_400, 86_400),
) as demo:
    gr.Markdown(
        """
        # Research Paper Q&A
        Upload academic PDFs, ask cited questions, or select two papers to compare.
        """
    )

    with gr.Row():
        # Left column: paper management.
        with gr.Column(scale=1, min_width=300):
            chunking_selector = gr.Radio(
                choices=list(CHUNKING_STRATEGIES),
                value=DEFAULT_CHUNKING_STRATEGY,
                label="Chunking strategy",
                info=(
                    "Naive is the current benchmark winner. Section-aware is "
                    "available as an experimental comparison."
                ),
            )
            runtime_markdown = gr.Markdown(runtime_status(DEFAULT_CHUNKING_STRATEGY))
            pdf_file = gr.File(
                label="Upload a PDF",
                file_types=[".pdf"],
                type="filepath",
            )
            ingest_button = gr.Button("Process and ingest", variant="primary")
            ingestion_status = gr.Markdown()

            paper_selector = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                max_choices=2,
                label="Papers for the next question",
                info="Choose one for Q&A, two for comparison, or none for automatic routing.",
            )
            delete_button = gr.Button("Delete selected from database", variant="stop")
            gr.Markdown("### Ingested papers")
            paper_list = gr.Markdown("_Loading papers..._")

        # Right column: question/answer conversation.
        with gr.Column(scale=3, min_width=500):
            chatbot = gr.Chatbot(
                label="Q&A session",
                height=480,
                placeholder="Ingest a paper, then ask a question.",
            )
            gr.Markdown(
                "_Questions are currently independent; earlier messages are "
                "displayed but are not passed back to the model._"
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a question about the selected paper(s)...",
                    show_label=False,
                    scale=5,
                )
                ask_button = gr.Button("Ask", variant="primary", scale=1)
            gr.Examples(
                examples=[
                    ["Which generator model does the RAG paper use?"],
                    ["Why are in-batch negatives efficient for training DPR?"],
                    ["Compare the passage collections used by DPR and RAG."],
                ],
                inputs=question_box,
                label="Example questions for the bundled papers",
            )

            with gr.Accordion("Retrieved sources", open=False):
                sources_markdown = gr.Markdown("_Sources appear after an answer._")
            clear_button = gr.ClearButton(
                [question_box, chatbot, sources_markdown],
                value="Clear Q&A session",
            )

    # Wire Python functions to UI events. Each output position matches the
    # corresponding value returned by the handler function.
    demo.load(
        refresh_papers,
        outputs=[paper_selector, paper_list],
    )
    ingest_button.click(
        ingest_pdf,
        inputs=[pdf_file, chunking_selector],
        outputs=[ingestion_status, paper_selector, paper_list],
    )
    chunking_selector.change(
        runtime_status,
        inputs=chunking_selector,
        outputs=runtime_markdown,
    )
    delete_button.click(
        remove_selected_papers,
        inputs=paper_selector,
        outputs=[ingestion_status, paper_selector, paper_list],
    )
    ask_button.click(
        answer_query,
        inputs=[question_box, chatbot, paper_selector],
        outputs=[question_box, chatbot, sources_markdown],
    )
    question_box.submit(
        answer_query,
        inputs=[question_box, chatbot, paper_selector],
        outputs=[question_box, chatbot, sources_markdown],
    )


def main() -> None:
    """Launch the Gradio application."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio link.",
    )
    args = parser.parse_args()
    share_demo = args.share or os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.queue(default_concurrency_limit=1).launch(
        share=share_demo,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
