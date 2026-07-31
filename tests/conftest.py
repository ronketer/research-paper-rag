"""
Shared pytest fixtures for the test suite.

All tests that need real PDF files should use the `papers_dir` or
`sample_pdf` fixtures instead of hardcoding paths. This ensures
the test suite works regardless of the current working directory,
and that adding/removing papers from papers/ doesn't break tests.
"""
from pathlib import Path

import pytest

# Canonical location of sample papers, relative to the project root
PAPERS_DIR = Path(__file__).parent.parent / "papers"


@pytest.fixture(scope="session")
def papers_dir() -> Path:
    """
    Return the Path to the papers/ directory.

    Fails fast with a clear message if the directory doesn't exist,
    rather than letting individual tests produce confusing FileNotFoundErrors.
    """
    assert PAPERS_DIR.exists(), (
        f"\npapers/ directory not found at: {PAPERS_DIR}\n"
        "Make sure the sample PDFs have been moved into the papers/ folder."
    )
    pdfs = list(PAPERS_DIR.glob("*.pdf"))
    assert pdfs, (
        f"\nNo PDF files found in {PAPERS_DIR}\n"
        "Add at least one .pdf to the papers/ folder before running tests."
    )
    return PAPERS_DIR


@pytest.fixture(scope="session")
def sample_pdf(papers_dir: Path) -> str:
    """
    Return the absolute path (as a string) to the first PDF in papers/.

    Using scope='session' so the file is resolved once per test run,
    not once per test function — loading PDFs is I/O-bound.
    """
    pdfs = sorted(papers_dir.glob("*.pdf"))
    return str(pdfs[0])
