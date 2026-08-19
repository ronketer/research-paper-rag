from types import SimpleNamespace

from src.application import rag_service


def test_ingest_document_loads_chunks_and_stores_paper(mocker):
    pages = [
        SimpleNamespace(paper_title="Demo Paper"),
    ]
    chunks = [object(), object()]

    mocker.patch(
        "src.application.rag_service.load_pdf",
        return_value=pages,
    )

    selected_chunker = mocker.Mock(return_value=chunks)
    mocker.patch.dict(
        rag_service.CHUNKERS,
        {"section_aware": selected_chunker},
    )

    mock_delete = mocker.patch(
        "src.application.rag_service.delete_paper"
    )
    mock_add = mocker.patch(
        "src.application.rag_service.add_paper"
    )

    result = rag_service.ingest_document(
        file_path="demo.pdf",
        chunking_strategy="section_aware",
    )

    selected_chunker.assert_called_once_with(pages)
    mock_delete.assert_called_once_with("Demo Paper")
    mock_add.assert_called_once_with("Demo Paper", chunks)

    assert result.paper_title == "Demo Paper"
    assert result.chunk_count == 2
