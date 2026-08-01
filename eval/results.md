# Evaluation Results

_Generated on 2026-08-01 by running:_

```bash
uv run python eval/evaluate.py --ingest --chunker both
```

The command completed locally without Langfuse credentials, an answer-generation
model, or an LLM judge. It rebuilt the Chroma index independently for each
strategy and evaluated all 30 benchmark questions.

## Chunker Comparison

| Metric | Naive | Section-aware | Delta |
|---|---:|---:|---:|
| Source-page Hit@K | 0.700 | 0.633 | -0.067 |
| Source-page Recall@K | 0.517 | 0.456 | -0.061 |
| Source-page Precision@K | 0.219 | 0.193 | -0.026 |
| Source-page F1@K | 0.298 | 0.260 | -0.038 |
| Source-paper Recall | 1.000 | 1.000 | +0.000 |
| Citation Presence | 1.000 | 1.000 | +0.000 |
| Citation Validity | 1.000 | 1.000 | +0.000 |

Delta is section-aware minus naive. Scores are macro averages across the 30
items. Higher is better.

## Run Details

| Strategy | DPR chunks | RAG chunks | REALM chunks |
|---|---:|---:|---:|
| Naive | 73 | 93 | 65 |
| Section-aware | 80 | 93 | 72 |

The run used four retrieved chunks for single-paper questions and three per paper
for comparisons. Retrieval used the local
`sentence-transformers/all-MiniLM-L6-v2` embedding model.

## Interpretation

Naive chunking performed better on every annotated source-page retrieval metric
in this run. Its source-page F1 was 0.298 versus 0.260 for section-aware chunking,
a difference of 0.038.

Source-paper recall was perfect for both strategies, so both reached the required
papers even when page-level ranking differed. Citation presence and validity were
also perfect because these are deterministic provenance checks: the runner emits
a citation manifest directly from retrieved page metadata. They do not measure
answer correctness or semantic quality.

## Reproduce

```bash
# Full benchmark:
uv run python eval/evaluate.py --ingest --chunker both

# Metric-arithmetic smoke test:
uv run python eval/evaluate.py --smoke-test
```
