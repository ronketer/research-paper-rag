from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.chains import answer_question, compare_papers
from src.chunker import Chunk, naive_chunk, section_aware_chunk
from src.loader import Page, load_pdf
from src.router import classify_query
from src.vectorstore import add_paper, delete_paper, list_papers

# ---------------------------------------------------------------------------
# Application result models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dependency types
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Default application dependencies
# ---------------------------------------------------------------------------

CHUNKERS: dict[str, ChunkerFn] = {
    "naive": naive_chunk,
    "section_aware": section_aware_chunk,
}


# ---------------------------------------------------------------------------
# Application service
# ---------------------------------------------------------------------------


class RAGService:
    """Coordinate the application's RAG use cases.

    The service owns application-level workflows such as:

    - document ingestion
    - single-paper Q&A
    - paper comparison
    - automatic query routing

    Infrastructure and implementation details are supplied as dependencies
    rather than being created inside the service.
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

        # Ingestion behaves as an upsert:
        # remove the old version before storing the new one.
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
        """Execute single-paper Q&A or multi-paper comparison."""

        selected = selected_papers or []

        # Explicit user selection has priority over automatic routing.
        if len(selected) == 2:
            paper_a, paper_b = selected
            return self._run_comparison(
                question,
                paper_a,
                paper_b,
            )

        if len(selected) == 1:
            return self._run_single_paper_query(
                question,
                selected[0],
            )

        # No manual selection: let the router determine the target papers.
        available_papers = self._list_papers()

        route = self._classify_query(
            question,
            available_papers,
        )

        routed_papers = route.get("papers", [])

        if (
            route.get("mode") == "comparison"
            and len(routed_papers) >= 2
        ):
            paper_a, paper_b = routed_papers[:2]

            return self._run_comparison(
                question,
                paper_a,
                paper_b,
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

    def _run_single_paper_query(
        self,
        question: str,
        paper: str | None,
    ) -> QueryResult:
        """Execute Q&A against one paper or the full paper collection."""

        result = self._answer_question(
            question,
            paper,
        )

        return QueryResult(
            answer=result["answer"],
            sources=result["sources"],
            source_heading=f"Paper: {paper or 'all papers'}",
        )

    def _run_comparison(
        self,
        question: str,
        paper_a: str,
        paper_b: str,
    ) -> QueryResult:
        """Execute a comparison between two papers."""

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


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def create_default_rag_service() -> RAGService:
    """Create the application's default RAG service.

    This function is the composition point where concrete implementations
    are connected to the application service.
    """

    return RAGService(
        load_pdf_fn=load_pdf,
        chunkers=CHUNKERS,
        add_paper_fn=add_paper,
        delete_paper_fn=delete_paper,
        list_papers_fn=list_papers,
        answer_question_fn=answer_question,
        compare_papers_fn=compare_papers,
        classify_query_fn=classify_query,
    )


# ---------------------------------------------------------------------------
# Compatibility API
#
# Keep these functions so app.py and existing callers do not need to change.
# Each call builds a lightweight service around the current dependencies.
# ---------------------------------------------------------------------------


def ingest_document(
    file_path: str,
    chunking_strategy: str,
) -> IngestionResult:
    """Ingest a document using the default application dependencies."""

    service = create_default_rag_service()

    return service.ingest_document(
        file_path=file_path,
        chunking_strategy=chunking_strategy,
    )


def run_query(
    question: str,
    selected_papers: list[str] | None = None,
) -> QueryResult:
    """Execute a query using the default application dependencies."""

    service = create_default_rag_service()

    return service.run_query(
        question=question,
        selected_papers=selected_papers,
    )
