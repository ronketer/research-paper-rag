# Research Paper RAG

A Python application for asking questions about academic PDFs and comparing two papers, with retrieved passages and page citations. It uses local embeddings and Chroma storage, with Ollama as the default answer provider.

The project includes a controlled retrieval experiment: the current regex-based section-aware chunker did not outperform the size-based baseline on the committed three-paper benchmark.

## Architecture

```text
PDF -> loader -> chunker -> Chroma
                            |
question -> routing -> retrieval -> answer generation -> citations
```

`app.py` is the Gradio interface. `src/paper_qa/bootstrap.py` constructs the application service and connects its dependencies.

| Package | Responsibility |
| --- | --- |
| `application/rag_service.py` | Ingestion, paper inventory and query workflows |
| `ingestion/` | PDF extraction, typed pages/chunks and the two chunkers |
| `infrastructure/vectorstore.py` | Persistent Chroma storage and metadata filtering |
| `routing/router.py` | Deterministic single-paper/comparison routing |
| `retrieval/service.py` | Shared retrieval policy for application and evaluation |
| `generation/` | Model calls, prompts and cited answer construction |
| `config/` | Provider and application configuration |

All package paths above are relative to `src/paper_qa/`. PDF extraction uses PyMuPDF, embeddings use sentence-transformers, and generation uses LangChain integrations.

## Setup

Requires Python 3.13+, uv, and an Ollama server for the default generation configuration.

```bash
git clone https://github.com/ronketer/research-paper-rag.git
cd research-paper-rag
uv sync --locked
cp .env.example .env
ollama pull qwen3:4b
uv run python app.py
```

In PowerShell, use `Copy-Item .env.example .env`. Start Ollama before asking questions. Open `http://localhost:7860`, ingest a PDF, then select one paper for Q&A or two for comparison. Each question is independent; conversational memory is not implemented.

`PAPER_QA_MODEL` selects the answer provider, for example `ollama:qwen3:4b`. A hosted provider requires its LangChain integration and credentials. Paper content sent to a hosted provider leaves the machine. Embedding/model downloads also require network access initially. `--share` creates a temporary public Gradio link; it is optional and does not constitute a hosted production deployment.

## Retrieval experiment

The [committed result report](eval/results.md), dated August 1, 2026, records macro averages across 30 annotated questions on the DPR, RAG and REALM papers. It uses `sentence-transformers/all-MiniLM-L6-v2`, four retrieved chunks for single-paper questions and three per paper for comparisons.

| Source-page metric | Naive | Section-aware |
| --- | ---: | ---: |
| Recall@K | 0.517 | 0.456 |
| Precision@K | 0.223 | 0.193 |
| F1@K | 0.299 | 0.260 |

These are results for this implementation and benchmark, not a general conclusion about section-aware chunking. Citation presence/validity checks use retrieved page metadata and do not measure answer correctness. Page annotations may omit other useful evidence.

```bash
# Metric arithmetic only, without papers or embeddings:
uv run python eval/evaluate.py --smoke-test

# Full retrieval comparison; rebuilds the local index for each strategy:
uv run python eval/evaluate.py --ingest --chunker both
```

The full comparison needs the sample PDFs and embedding model, but no answer-generation model or LLM judge. See [benchmark annotations](eval/benchmark.json) and [run details](eval/results.md).

## Tests

```bash
uv run pytest tests/ -v
```

Tests cover loading, chunking, routing, retrieval allocation, generation boundaries, vector storage, application workflows and Gradio handlers. Several boundaries use mocks; loader tests use the sample PDFs. A successful test run does not establish live-provider answer quality. No passed-test total is asserted here without a recorded run for the current revision.

## Limitations

- Text extraction and regex headings can fail on unfamiliar PDF layouts; scanned-document OCR is not implemented.
- Routing uses keyword and paper-name rules rather than learned intent recognition.
- The benchmark has only three papers and 30 questions. It evaluates retrieval and citation provenance, not semantic answer quality.
- Persistent storage and single-process UI behavior are intended for a local demo. Authentication and tenant isolation are not established.
- Local inference is the default configuration, but hosted providers and public Gradio sharing are supported options with different data-exposure boundaries.

## License and source material

Project code is MIT-licensed; see [LICENSE](LICENSE). The sample papers remain the work of their authors and publishers. The code license does not grant rights to redistribute those PDFs.
