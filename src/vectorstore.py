import math

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.chunker import Chunk

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"


_embedding_model: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the cached local embedding model (no API key needed)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedding_model


def get_vectorstore() -> Chroma:
    """Return the cached persistent ChromaDB instance."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=get_embedding_model(),
        )
    return _vectorstore


def add_paper(paper_title: str, chunks: list[Chunk]) -> None:
    """
    Embed and store chunks for a paper.

    Steps:
    1. Convert Chunk objects to LangChain Documents (text + metadata dict)
    2. Add to ChromaDB with the embedding model
    """
    db = get_vectorstore()
    documents = []
    for chunk in chunks:
        doc_metadata ={
            "paper_title": paper_title,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "section": chunk.section
        }

        doc = Document(page_content=chunk.text, metadata=doc_metadata)
        documents.append(doc)

    db.add_documents(documents)


def retrieve(query: str, paper_filter: str | list[str] | None = None, k: int = 5) -> list[Document]:
    """
    Find the k most relevant chunks for a query.

    Args:
        query: the user's question
        paper_filter: limit search to specific paper(s), or None for all
        k: number of chunks to return

    Returns:
        List of LangChain Document objects with metadata
    """
    db = get_vectorstore()

    if not paper_filter:
        return db.similarity_search(query, k=k)

    if isinstance(paper_filter, str):
        filter_dict = {"paper_title": paper_filter}
        return db.similarity_search(query, k=k, filter=filter_dict)


    # Distribute k evenly across papers so no single paper dominates the results.
    k_per_paper = max(1, math.ceil(k / len(paper_filter)))
    all_results = []
    for title in paper_filter:
        filter_dict = {"paper_title": title}
        results = db.similarity_search(query, k=k_per_paper, filter=filter_dict)
        all_results.extend(results)

    return all_results[:k]

def list_papers() -> list[str]:
    """Return names of all ingested papers."""
    db = get_vectorstore()
    results = db.get(include=["metadatas"])


    titles = set()
    for m in results["metadatas"]:
        if m and "paper_title" in m:
            titles.add(m["paper_title"])


    return sorted(titles)



def delete_paper(paper_title: str) -> None:
    """Remove all chunks belonging to a paper from the store."""
    db = get_vectorstore()
    db.delete(where={"paper_title": paper_title})
