import math

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from paper_qa.config.settings import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
)
from paper_qa.ingestion.models import Chunk

_embedding_model: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the cached local embedding model."""

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embedding_model


def get_vectorstore() -> Chroma:
    """Return the cached persistent Chroma instance."""

    global _vectorstore

    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=str(
                CHROMA_PERSIST_DIR
            ),
            embedding_function=get_embedding_model(),
        )

    return _vectorstore


def add_paper(
    paper_title: str,
    chunks: list[Chunk],
) -> None:
    """Persist embedded paper chunks."""

    db = get_vectorstore()

    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "paper_title": paper_title,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
            },
        )
        for chunk in chunks
    ]

    db.add_documents(documents)


def retrieve(
    query: str,
    paper_filter: str | list[str] | None = None,
    k: int = 5,
) -> list[Document]:
    """Perform similarity search against Chroma."""

    db = get_vectorstore()

    if not paper_filter:
        return db.similarity_search(
            query,
            k=k,
        )

    if isinstance(paper_filter, str):
        return db.similarity_search(
            query,
            k=k,
            filter={
                "paper_title": paper_filter,
            },
        )

    k_per_paper = max(
        1,
        math.ceil(k / len(paper_filter)),
    )

    results: list[Document] = []

    for title in paper_filter:
        paper_results = db.similarity_search(
            query,
            k=k_per_paper,
            filter={
                "paper_title": title,
            },
        )

        results.extend(paper_results)

    return results[:k]


def list_papers() -> list[str]:
    """Return unique ingested paper titles."""

    db = get_vectorstore()

    results = db.get(
        include=["metadatas"]
    )

    titles = {
        metadata["paper_title"]
        for metadata in results["metadatas"]
        if metadata
        and "paper_title" in metadata
    }

    return sorted(titles)


def delete_paper(
    paper_title: str,
) -> None:
    """Delete all chunks belonging to a paper."""

    db = get_vectorstore()

    db.delete(
        where={
            "paper_title": paper_title,
        }
    )
