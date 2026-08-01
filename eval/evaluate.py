"""Run the deterministic paper retrieval benchmark locally.

No Langfuse credentials, answer-generation model, or LLM judge are required.

Examples:
    uv run python eval/evaluate.py --smoke-test
    uv run python eval/evaluate.py --ingest --chunker section-aware
    uv run python eval/evaluate.py --ingest --chunker both
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import naive_chunk, section_aware_chunk
from src.loader import load_pdf
from src.router import classify_query
from src.vectorstore import add_paper, delete_paper, list_papers, retrieve

PAPERS_DIR = PROJECT_ROOT / "papers"
BENCHMARK_FILE = PROJECT_ROOT / "eval" / "benchmark.json"

Chunker = Callable[..., list]
CHUNKERS: dict[str, Chunker] = {
    "naive": naive_chunk,
    "section-aware": section_aware_chunk,
}


class BenchmarkItem(BaseModel):
    """Schema for one version-controlled benchmark item."""

    id: str
    question: str
    ground_truth: str
    type: Literal["factual", "reasoning", "comparison"]
    source_papers: list[str] = Field(min_length=1)
    source_pages: dict[str, list[int]] = Field(min_length=1)


EXPECTED_TYPE_COUNTS = {"factual": 10, "reasoning": 10, "comparison": 10}


@dataclass(frozen=True)
class Metric:
    """One deterministic item-level score."""

    name: str
    value: float
    detail: str = ""


def load_benchmark() -> list[BenchmarkItem]:
    """Load and validate the balanced local benchmark."""
    with BENCHMARK_FILE.open("r", encoding="utf-8") as file:
        raw_benchmark = json.load(file)
    if not isinstance(raw_benchmark, list):
        raise ValueError("Benchmark must be a JSON array of items")

    benchmark: list[BenchmarkItem] = []
    for index, raw_item in enumerate(raw_benchmark):
        try:
            benchmark.append(BenchmarkItem.model_validate(raw_item))
        except Exception as error:
            raise ValueError(f"Invalid benchmark item {index}: {error}") from error

    ids = [item.id for item in benchmark]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark item ids must be unique")
    type_counts = Counter(item.type for item in benchmark)
    if dict(type_counts) != EXPECTED_TYPE_COUNTS:
        raise ValueError(
            "Benchmark must contain exactly 10 factual, 10 reasoning, and "
            f"10 comparison items; found {dict(type_counts)}"
        )
    return benchmark


def ingest_papers(chunker_name: str) -> None:
    """Rebuild the local index from benchmark PDFs with one chunking strategy."""
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PAPERS_DIR}")

    # --ingest denotes a clean benchmark rebuild. Clearing all indexed papers
    # prevents stale chunks from contaminating the chunker comparison.
    for paper_title in list_papers():
        delete_paper(paper_title)

    chunker = CHUNKERS[chunker_name]
    print(f"\nIngesting {len(pdf_files)} papers with '{chunker_name}' chunking:")
    for pdf in pdf_files:
        pages = load_pdf(str(pdf))
        if not pages:
            print(f"  [SKIP] No text extracted from {pdf.name}")
            continue
        paper_title = pages[0].paper_title
        chunks = chunker(pages)
        add_paper(paper_title, chunks)
        print(f"  [OK] {paper_title}: {len(chunks)} chunks")


def _page(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def retrieve_for_question(question: str) -> dict[str, Any]:
    """Run production-style deterministic routing and retrieval."""
    route = classify_query(question, list_papers())
    if route["mode"] == "comparison" and len(route["papers"]) >= 2:
        docs = retrieve(question, paper_filter=route["papers"][0], k=3)
        docs += retrieve(question, paper_filter=route["papers"][1], k=3)
    else:
        target = route["papers"][0] if route["papers"] else None
        docs = retrieve(question, paper_filter=target, k=4)

    sources = [
        {
            "paper": doc.metadata.get("paper_title", "Unknown"),
            "page": doc.metadata.get("page_number", "Unknown"),
            "section": doc.metadata.get("section", "Unknown"),
        }
        for doc in docs
    ]
    pages = [page for source in sources if (page := _page(source["page"])) is not None]
    return {
        # This citation manifest tests metadata provenance, not answer quality.
        "answer": "Retrieved evidence: " + " ".join(f"[p. {page}]" for page in pages),
        "retrieved_sources": sources,
        "route": route,
    }


def _normalise_paper(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _expected_source_pairs(item: BenchmarkItem) -> set[tuple[str, int]]:
    return {
        (_normalise_paper(paper), page)
        for paper, pages in item.source_pages.items()
        for value in pages
        if (page := _page(value)) is not None
    }


def _retrieved_source_pairs(output: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (_normalise_paper(source.get("paper", "")), page)
        for source in output.get("retrieved_sources", [])
        if (page := _page(source.get("page"))) is not None
    }


def source_page_hit_at_k(item: BenchmarkItem, output: dict[str, Any]) -> Metric:
    expected = _expected_source_pairs(item)
    retrieved = _retrieved_source_pairs(output)
    matched = len(expected & retrieved)
    return Metric("source_page_hit_at_k", float(bool(matched)), f"matched={matched}")


def source_page_recall_at_k(item: BenchmarkItem, output: dict[str, Any]) -> Metric:
    expected = _expected_source_pairs(item)
    retrieved = _retrieved_source_pairs(output)
    matched = len(expected & retrieved)
    score = matched / len(expected) if expected else 0.0
    return Metric("source_page_recall_at_k", score, f"matched={matched}, expected={len(expected)}")


def source_page_precision_at_k(item: BenchmarkItem, output: dict[str, Any]) -> Metric:
    expected = _expected_source_pairs(item)
    retrieved = _retrieved_source_pairs(output)
    matched = len(expected & retrieved)
    score = matched / len(retrieved) if retrieved else 0.0
    return Metric(
        "source_page_precision_at_k",
        score,
        f"matched={matched}, retrieved={len(retrieved)}",
    )


def source_page_f1_at_k(item: BenchmarkItem, output: dict[str, Any]) -> Metric:
    expected = _expected_source_pairs(item)
    retrieved = _retrieved_source_pairs(output)
    matched = len(expected & retrieved)
    precision = matched / len(retrieved) if retrieved else 0.0
    recall = matched / len(expected) if expected else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metric("source_page_f1_at_k", score)


def source_paper_recall(item: BenchmarkItem, output: dict[str, Any]) -> Metric:
    expected = {paper for paper, _ in _expected_source_pairs(item)}
    retrieved = {paper for paper, _ in _retrieved_source_pairs(output)}
    matched = len(expected & retrieved)
    score = matched / len(expected) if expected else 0.0
    return Metric("source_paper_recall", score, f"matched={matched}, expected={len(expected)}")


def citation_presence(_: BenchmarkItem, output: dict[str, Any]) -> Metric:
    present = bool(re.search(r"\[p\.\s*\d+\]", output.get("answer", ""), re.I))
    return Metric("citation_presence", float(present))


def citation_validity(_: BenchmarkItem, output: dict[str, Any]) -> Metric:
    cited_pages = [
        int(value)
        for value in re.findall(r"\[p\.\s*(\d+)\]", output.get("answer", ""), re.I)
    ]
    retrieved_pages = {
        page
        for source in output.get("retrieved_sources", [])
        if (page := _page(source.get("page"))) is not None
    }
    valid = sum(page in retrieved_pages for page in cited_pages)
    score = valid / len(cited_pages) if cited_pages else 0.0
    return Metric("citation_validity", score, f"valid={valid}, citations={len(cited_pages)}")


EVALUATORS: tuple[Callable[[BenchmarkItem, dict[str, Any]], Metric], ...] = (
    source_page_hit_at_k,
    source_page_recall_at_k,
    source_page_precision_at_k,
    source_page_f1_at_k,
    source_paper_recall,
    citation_presence,
    citation_validity,
)

METRIC_LABELS = {
    "source_page_hit_at_k": "Source-page Hit@K",
    "source_page_recall_at_k": "Source-page Recall@K",
    "source_page_precision_at_k": "Source-page Precision@K",
    "source_page_f1_at_k": "Source-page F1@K",
    "source_paper_recall": "Source-paper Recall",
    "citation_presence": "Citation Presence",
    "citation_validity": "Citation Validity",
}


def evaluate_chunker(chunker_name: str, benchmark: list[BenchmarkItem]) -> dict[str, float]:
    """Run all items and return macro-average deterministic scores."""
    totals = {evaluator.__name__: 0.0 for evaluator in EVALUATORS}
    print(f"Evaluating {len(benchmark)} benchmark questions:")
    for index, item in enumerate(benchmark, start=1):
        output = retrieve_for_question(item.question)
        for evaluator in EVALUATORS:
            result = evaluator(item, output)
            totals[result.name] += result.value
        print(f"  [{index:02d}/{len(benchmark)}] {item.id}")
    return {name: total / len(benchmark) for name, total in totals.items()}


def print_results(results: dict[str, dict[str, float]]) -> None:
    """Print a Markdown table suitable for eval/results.md."""
    names = list(results)
    print("\n## Chunker Comparison\n")
    if len(names) == 2:
        print("| Metric | Naive | Section-aware | Delta |")
        print("|---|---:|---:|---:|")
        naive = results["naive"]
        section = results["section-aware"]
        for metric_name, label in METRIC_LABELS.items():
            delta = section[metric_name] - naive[metric_name]
            print(f"| {label} | {naive[metric_name]:.3f} | {section[metric_name]:.3f} | {delta:+.3f} |")
    else:
        chunker_name = names[0]
        print(f"| Metric | {chunker_name} |")
        print("|---|---:|")
        for metric_name, label in METRIC_LABELS.items():
            print(f"| {label} | {results[chunker_name][metric_name]:.3f} |")


def run_smoke_test() -> None:
    """Exercise deterministic metric arithmetic without loading embeddings."""
    item = BenchmarkItem(
        id="smoke",
        question="Smoke test",
        ground_truth="",
        type="factual",
        source_papers=["Paper A"],
        source_pages={"Paper A": [1, 2]},
    )
    output = {
        "answer": "Evidence [p. 1] and invalid citation [p. 9]",
        "retrieved_sources": [
            {"paper": "Paper A", "page": 1},
            {"paper": "Paper B", "page": 3},
        ],
    }
    scores = {
        evaluator.__name__: evaluator(item, output).value
        for evaluator in (
            source_page_recall_at_k,
            source_page_precision_at_k,
            source_page_f1_at_k,
            citation_validity,
        )
    }
    if any(score != 0.5 for score in scores.values()):
        raise SystemExit(f"Deterministic smoke test failed: {scores}")
    print(f"[OK] Deterministic metrics smoke test passed: {scores}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest", action="store_true", help="Rebuild the local index before each run.")
    parser.add_argument(
        "--chunker",
        choices=["naive", "section-aware", "both"],
        default="section-aware",
        help="Chunking strategy to evaluate (default: section-aware).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate metric arithmetic without loading papers or embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
        return
    if args.chunker == "both" and not args.ingest:
        raise SystemExit("--chunker both requires --ingest to isolate the strategies.")

    benchmark = load_benchmark()
    chunker_names = list(CHUNKERS) if args.chunker == "both" else [args.chunker]
    results: dict[str, dict[str, float]] = {}
    for chunker_name in chunker_names:
        if args.ingest:
            ingest_papers(chunker_name)
        elif not list_papers():
            raise SystemExit("The local index is empty. Re-run with --ingest.")
        results[chunker_name] = evaluate_chunker(chunker_name, benchmark)
    print_results(results)


if __name__ == "__main__":
    main()
