from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    """A single page extracted from a source document."""

    page_number: int
    text: str
    paper_title: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a paper with provenance metadata."""

    text: str
    paper_title: str
    section: str
    page_number: int
    chunk_index: int
