"""Tests for src/router.py"""

import pytest

from src.router import _extract_mentioned_papers, classify_query

# ---------------------------------------------------------------------------
# Tests for _extract_mentioned_papers
# ---------------------------------------------------------------------------

def test_extract_exact_match():
    """Should find papers when their exact title is in the query (case-insensitive)."""
    available = ["Attention Is All You Need", "BERT", "GPT-3"]
    question = "How does attention is all you need work?"

    mentioned = _extract_mentioned_papers(question, available)
    assert mentioned == ["Attention Is All You Need"]


def test_extract_fallback_match_extensions():
    """Should find papers by stripping file extensions."""
    available = ["paper1.pdf", "paper2.txt", "paper3.md"]
    question = "What is the conclusion of paper1?"

    mentioned = _extract_mentioned_papers(question, available)
    assert mentioned == ["paper1.pdf"]


def test_extract_fallback_match_underscores_and_dashes():
    """Should find papers by converting underscores and dashes to spaces."""
    available = ["my_awesome_paper.pdf", "another-great-study.txt"]
    question = "Tell me about my awesome paper and another great study."

    mentioned = _extract_mentioned_papers(question, available)
    assert mentioned == ["another-great-study.txt", "my_awesome_paper.pdf"]


def test_extract_short_titles_ignored_in_fallback():
    """Should ignore very short cleaned titles (<= 4 chars) to prevent false positives."""
    # "cat.pdf" cleans to "cat" (3 chars). It shouldn't match the word "cat" in a sentence.
    available = ["cat.pdf", "dog.txt"]
    question = "My cat is chasing a dog."

    mentioned = _extract_mentioned_papers(question, available)
    # It shouldn't match either because "cat" and "dog" are <= 4 chars
    assert mentioned == []


def test_extract_multiple_papers():
    """Should return multiple papers sorted alphabetically and deduplicated."""
    available = ["Paper B", "Paper A", "Paper C"]
    question = "Compare paper c with paper a, and mention paper a again."

    mentioned = _extract_mentioned_papers(question, available)
    assert mentioned == ["Paper A", "Paper C"]


# ---------------------------------------------------------------------------
# Tests for classify_query
# ---------------------------------------------------------------------------

def test_classify_single_paper_explicit():
    """If exactly one paper is mentioned and no comparison keywords, return single_paper mode."""
    available = ["Paper A", "Paper B"]
    question = "What is the main finding in Paper A?"

    result = classify_query(question, available)

    assert result["mode"] == "single_paper"
    assert result["papers"] == ["Paper A"]


def test_classify_single_paper_fallback():
    """If no papers and no keywords are found, fallback to single_paper with empty list."""
    available = ["Paper A", "Paper B"]
    question = "What is the meaning of life?"

    result = classify_query(question, available)

    assert result["mode"] == "single_paper"
    assert result["papers"] == []


def test_classify_comparison_explicit_two_papers():
    """If two papers are mentioned (even without keywords), trigger comparison mode."""
    available = ["Paper A", "Paper B", "Paper C"]
    question = "Summarize Paper A and Paper B."

    result = classify_query(question, available)

    assert result["mode"] == "comparison"
    assert result["papers"] == ["Paper A", "Paper B"]


def test_classify_comparison_with_keyword_two_papers():
    """Comparison keyword with two papers returns those two papers."""
    available = ["Paper A", "Paper B", "Paper C"]
    question = "Compare Paper C and Paper A."

    result = classify_query(question, available)

    assert result["mode"] == "comparison"
    assert result["papers"] == ["Paper A", "Paper C"]


def test_classify_comparison_with_keyword_one_paper():
    """Comparison keyword with 1 paper -> backfill up to 2 papers from available."""
    available = ["Paper A", "Paper B", "Paper C"]
    question = "How does Paper B compare to others?"

    result = classify_query(question, available)

    assert result["mode"] == "comparison"
    assert result["papers"] == ["Paper B", "Paper A"]


def test_classify_comparison_with_keyword_zero_papers():
    """Comparison keyword with 0 papers -> pick the first 2 available papers."""
    available = ["Paper A", "Paper B", "Paper C"]
    question = "What is the difference between them?"

    result = classify_query(question, available)

    assert result["mode"] == "comparison"
    assert result["papers"] == ["Paper A", "Paper B"]


def test_classify_wildcard_comparison_keyword():
    """Should correctly process wildcard keywords like 'how does * differ'."""
    available = ["Paper A", "Paper B"]
    # The '*' in 'how does * differ' should match 'this'
    question = "how does this differ?"

    result = classify_query(question, available)

    assert result["mode"] == "comparison"
    assert len(result["papers"]) == 2
