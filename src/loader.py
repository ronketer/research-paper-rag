import pymupdf4llm
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PaperContent:
    title: str
    markdown: str        # one big Markdown string — YOUR code splits it
    page_count: int

def load_pdf(file_path: str) -> PaperContent:
    """
    Extract PDF to Markdown. That's ALL this function does.
    Chunking happens in chunker.py — NOT here.
    """
    md_text = pymupdf4llm.to_markdown(
        file_path,
        page_chunks=False,       
        header=False,            
        footer=False,
    )
    title = _extract_title(md_text, file_path)

    return PaperContent(title=title, markdown=md_text, page_count=_count_pages(md_text))



# TODO: fix page counting.

def _count_pages(md_text: str) -> int:
    """Count pages in Markdown text."""
    return md_text.count("\f") + 1 

def _extract_title(md_text: str, file_path: str) -> str:
    """Extract title from first # heading, fall back to filename."""
    for line in md_text.split("\n"):
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return Path(file_path).stem.replace("-", " ").replace("_", " ").title()
