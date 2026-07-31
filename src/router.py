import re
from typing import Any, Dict, List

COMPARISON_KEYWORDS = [
    "compare", "difference", "differ", "vs", "versus",
    "contrast", "how does * differ", "similarities",
    "both papers", "each paper"
]


def _extract_mentioned_papers(question: str, available_papers: list[str]) -> list[str]:
    """
    Helper function to scan the user query text and identify which
    known, ingested papers are being requested.
    """
    mentioned = []
    lowered_question = question.lower()

    for paper in available_papers:
        # Lowercase for loose text matching
        paper_lower = paper.lower()

        # Scenario A: Precise check if the paper title is explicitly inside the text
        if paper_lower in lowered_question:
            mentioned.append(paper)
            continue

        # Scenario B: Fallback check for common filename/abbreviation variations
        # (e.g., stripping extensions like .pdf, replacing underscores/dashes with spaces)
        clean_title = re.sub(r'\.(pdf|txt|md)$', '', paper_lower)
        clean_title = clean_title.replace("_", " ").replace("-", " ")

        if len(clean_title) > 4 and clean_title in lowered_question:
            mentioned.append(paper)

    return sorted(list(set(mentioned)))


def classify_query(question: str, available_papers: list[str]) -> dict:
    """
    Determine if a question is single-paper or comparison.

    Returns:
        {
            "mode": "single_paper" | "comparison",
            "papers": ["paper_a"] or ["paper_a", "paper_b"]
        }
    """
    lowered_question = question.lower()

    # 1. Check for comparison keywords
    has_comparison_keyword = False
    for keyword in COMPARISON_KEYWORDS:
        # Check simple keyword inclusion
        if keyword in lowered_question:
            has_comparison_keyword = True
            break
        # Dynamic wildcard matching for phrases like "how does * differ"
        if "*" in keyword:
            pattern = keyword.replace("*", ".*")
            if re.search(pattern, lowered_question):
                has_comparison_keyword = True
                break

    # 2. Check if multiple paper names are mentioned
    mentioned_papers = _extract_mentioned_papers(question, available_papers)

    # 3. If comparison keywords OR multiple papers detected -> comparison mode
    # Requires at least two papers to execute a real comparison. If only 2 papers total
    # exist in the database and they are inferred, we default to comparing them.
    if has_comparison_keyword or len(mentioned_papers) >= 2:
        # If the user said "compare" but named 0 or 1 papers, grab the first two available
        # to ensure comparison logic doesn't crash, or fallback to all papers.
        target_papers = mentioned_papers
        if len(target_papers) < 2:
            # If the user named 1 paper, combine it with whatever else is available
            target_papers = mentioned_papers + [p for p in available_papers if p not in mentioned_papers]
            # Limit safely to 2 papers total for your COMPARISON_PROMPT layout
            target_papers = target_papers[:2]

        return {
            "mode": "comparison",
            "papers": target_papers
        }

    # 4. If exactly one paper is mentioned -> single_paper mode
    if len(mentioned_papers) == 1:
        return {
            "mode": "single_paper",
            "papers": mentioned_papers
        }

    # 5. If no specific paper was mentioned -> default fallback
    # Routes to search across all available papers using single_paper mode
    # (Your vectorstore retrieve() handles paper_filter=None beautifully!)
    return {
        "mode": "single_paper",
        "papers": []  # Empty array signals downstream to search everything
    }
