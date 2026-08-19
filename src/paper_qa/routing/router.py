import re

COMPARISON_KEYWORDS = [
    "compare",
    "difference",
    "differ",
    "vs",
    "versus",
    "contrast",
    "how does * differ",
    "similarities",
    "both papers",
    "each paper",
]


def _extract_mentioned_papers(
    question: str,
    available_papers: list[str],
) -> list[str]:
    """Identify known paper titles mentioned in a query."""

    mentioned: list[str] = []
    lowered_question = question.lower()

    for paper in available_papers:
        paper_lower = paper.lower()

        if paper_lower in lowered_question:
            mentioned.append(paper)
            continue

        clean_title = re.sub(
            r"\.(pdf|txt|md)$",
            "",
            paper_lower,
        )

        clean_title = (
            clean_title
            .replace("_", " ")
            .replace("-", " ")
        )

        if (
            len(clean_title) > 4
            and clean_title in lowered_question
        ):
            mentioned.append(paper)

    return sorted(set(mentioned))


def classify_query(
    question: str,
    available_papers: list[str],
) -> dict:
    """Determine single-paper versus comparison routing."""

    lowered_question = question.lower()

    has_comparison_keyword = False

    for keyword in COMPARISON_KEYWORDS:
        if keyword in lowered_question:
            has_comparison_keyword = True
            break

        if "*" in keyword:
            pattern = keyword.replace("*", ".*")

            if re.search(
                pattern,
                lowered_question,
            ):
                has_comparison_keyword = True
                break

    mentioned_papers = _extract_mentioned_papers(
        question,
        available_papers,
    )

    if (
        has_comparison_keyword
        or len(mentioned_papers) >= 2
    ):
        target_papers = mentioned_papers

        if len(target_papers) < 2:
            target_papers = (
                mentioned_papers
                + [
                    paper
                    for paper in available_papers
                    if paper not in mentioned_papers
                ]
            )

            target_papers = target_papers[:2]

        return {
            "mode": "comparison",
            "papers": target_papers,
        }

    if len(mentioned_papers) == 1:
        return {
            "mode": "single_paper",
            "papers": mentioned_papers,
        }

    return {
        "mode": "single_paper",
        "papers": [],
    }
