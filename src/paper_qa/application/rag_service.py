from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from paper_qa.ingestion.models import Chunk, Page


@dataclass(frozen=True)
class IngestionResult:
    """Result returned after successfully ingesting a paper."""

    paper_title: str
    chunk_count: int


@dataclass(frozen=True)
class QueryResult:
    """Result returned after executing a Q&A or comparison request."""

    answer: str
    sources: list[dict[str, Any]]
    source_heading: str


PageLoader = Callable[[str], list[Page]]
ChunkerFn = Callable[[list[Page]], list[Chunk]]

AddPaperFn = Callable[[str, list[Chunk]], None]
DeletePaperFn = Callable[[str], None]
ListPapersFn = Callable[[], list[str]]

AnswerQuestionFn = Callable[
    [str, str | None],
    dict[str, Any],
]

ComparePapersFn = Callable[
    [str, str, str],
    dict[str, Any],
]

ClassifyQueryFn = Callable[
    [str, list[str]],
    dict[str, Any],
]


class RAGService:
    """Coordinate application-level RAG use cases.

    The application service owns workflow decisions but does not know
    which concrete PDF loader, vector database, router, or LLM provider
    implements those capabilities.
    """

    def __init__(
        self,
        *,
        load_pdf_fn: PageLoader,
        chunkers: Mapping[str, ChunkerFn],
        add_paper_fn: AddPaperFn,
        delete_paper_fn: DeletePaperFn,
        list_papers_fn: ListPapersFn,
        answer_question_fn: AnswerQuestionFn,
        compare_papers_fn: ComparePapersFn,
        classify_query_fn: ClassifyQueryFn,
    ) -> None:
        self._load_pdf = load_pdf_fn
        self._chunkers = dict(chunkers)

        self._add_paper = add_paper_fn
        self._delete_paper = delete_paper_fn
        self._list_papers = list_papers_fn

        self._answer_question = answer_question_fn
        self._compare_papers = compare_papers_fn
        self._classify_query = classify_query_fn

    def ingest_document(
        self,
        file_path: str,
        chunking_strategy: str,
    ) -> IngestionResult:
        """Load, chunk, and persist a PDF."""

        pages = self._load_pdf(file_path)

        if not pages:
            raise ValueError(
                "No text could be extracted from this PDF."
            )

        chunker = self._chunkers.get(chunking_strategy)

        if chunker is None:
            raise ValueError(
                f"Unknown chunking strategy: {chunking_strategy}"
            )

        title = pages[0].paper_title
        chunks = chunker(pages)

        # Re-ingestion replaces the previous stored version.
        self._delete_paper(title)
        self._add_paper(title, chunks)

        return IngestionResult(
            paper_title=title,
            chunk_count=len(chunks),
        )

    def run_query(
        self,
        question: str,
        selected_papers: list[str] | None = None,
    ) -> QueryResult:
        """Execute Q&A or paper comparison."""

        selected = selected_papers or []

        # Manual UI selection has priority over automatic routing.
        if len(selected) == 2:
            return self._run_comparison(
                question,
                selected[0],
                selected[1],
            )

        if len(selected) == 1:
            return self._run_single_paper_query(
                question,
                selected[0],
            )

        available_papers = self._list_papers()

        route = self._classify_query(
            question,
            available_papers,
        )

        routed_papers = route.get(
            "papers",
            [],
        )

        if (
            route.get("mode") == "comparison"
            and len(routed_papers) >= 2
        ):
            return self._run_comparison(
                question,
                routed_papers[0],
                routed_papers[1],
            )

        paper = (
            routed_papers[0]
            if routed_papers
            else None
        )

        return self._run_single_paper_query(
            question,
            paper,
        )

    def list_papers(self) -> list[str]:
        """Return currently ingested paper titles."""

        return self._list_papers()

    def delete_paper(
        self,
        paper_title: str,
    ) -> None:
        """Delete one ingested paper."""

        self._delete_paper(paper_title)

    def _run_single_paper_query(
        self,
        question: str,
        paper: str | None,
    ) -> QueryResult:
        """Execute Q&A against one paper or all papers."""

        result = self._answer_question(
            question,
            paper,
        )

        return QueryResult(
            answer=result["answer"],
            sources=result["sources"],
            source_heading=(
                f"Paper: {paper or 'all papers'}"
            ),
        )

    def _run_comparison(
        self,
        question: str,
        paper_a: str,
        paper_b: str,
    ) -> QueryResult:
        """Execute a two-paper comparison."""

        result = self._compare_papers(
            question,
            paper_a,
            paper_b,
        )

        return QueryResult(
            answer=result["comparison"],
            sources=(
                result["sources_a"]
                + result["sources_b"]
            ),
            source_heading=(
                f"Comparison: {paper_a} vs {paper_b}"
            ),
        )
