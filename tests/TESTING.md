# Evaluation Guide

## Regression tests

With the project dependencies installed, run:

```bash
python -m unittest discover -s tests -p 'test_regressions.py' -v
```

These tests cover context-window defaults and upload cleanup. Store operations,
extraction, and reranking are mocked; running services and model downloads are not required.

Retrieval quality is measured against a golden dataset of question/chunk pairs.
Each pair names the chunk that *should* come back for a question, so a search run
can be scored automatically.

Everything below assumes the stack is up and the corpus is ingested:

```bash
docker-compose up -d          # wait 30-60s for health checks
curl -s localhost:8000/documents | jq '.count'
```

## 1. Build a golden dataset

`generate_golden_dataset.py` samples chunks from Weaviate and asks Claude to write
questions answerable from each one. Output is JSONL, one
`{question, file_id, filename, chunk_index, page_number, source_text}` per line.

```bash
python tests/generate_golden_dataset.py --chunks-per-doc 3 --questions-per-chunk 1
```

| Option | Default | Meaning |
|---|---|---|
| `--chunks-per-doc` | 3 | Chunks sampled per document |
| `--questions-per-chunk` | 1 | Questions generated per chunk |
| `--output` | `tests/golden_dataset.jsonl` | Where to write |
| `--model` | `claude-haiku-4-5-20251001` | Generation model |
| `--review` | off | Confirm or skip each pair interactively |
| `--dry-run` | off | Show sampled chunks without calling Claude |

Requires `HYDE_API_KEY` in `.env`. Generation is semi-automatic: review the output
and delete or edit weak questions before trusting the scores.

## 2. Score retrieval

`evaluate_retrieval.py` replays every question through the search API and reports
where the expected chunk landed.

```bash
python tests/evaluate_retrieval.py --k 5 --output tests/results/run.json
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | `tests/golden_dataset.jsonl` | Dataset to replay |
| `--api-url` | `http://localhost:8000` | API base URL |
| `--k` | 5 | Cutoff for Recall@K |
| `--modes` | all | `vector`, `keywords`, `hybrid`, `hybrid_hyde` |
| `--output` | none | Save metrics as JSON |

Reported metrics:

- **Recall@K** — the expected chunk appears in the top K.
- **DocRecall@2** — the right *document* appears in the top 2, a softer target that
  tolerates the answer landing in a neighbouring chunk.
- **MRR** — mean reciprocal rank of the expected chunk.
- **P@1** — the expected chunk is the very first result.
- **Avg ms** — mean latency per query.

A hit requires both `file_id` and `chunk_index` to match.


## Manual API checks

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/upload -F "file=@tests/attention3.pdf"
curl -s localhost:8000/task/<task_id>          # poll until SUCCESS
curl -s localhost:8000/documents

curl -s -X POST localhost:8000/search/vector \
  -H 'Content-Type: application/json' -d '{"q":"scaled dot-product attention","limit":5}'

curl -s -X POST localhost:8000/search/double \
  -H 'Content-Type: application/json' \
  -d '{"q":"scaled dot-product attention","limit":5,"use_hyde":false}'
```

Every hybrid response carries `retrieval_mode`, naming the strategy that actually ran:

| `retrieval_mode` | Meaning | Diagnostics returned |
|---|---|---|
| `rrf` | Weaviate vector + Elasticsearch BM25, fused by RRF | `vector_count`, `keyword_count`, `unique_count`, `rrf_k` |
| `weaviate_hybrid` | Weaviate's built-in hybrid, blended by alpha | `alpha`, `candidate_count` |

The two are different algorithms, so a score is only comparable to another run in the
same mode. Always record `retrieval_mode` alongside your metrics.

```bash
curl -s -X POST localhost:8000/search/double \
  -H 'Content-Type: application/json' -d '{"q":"scaled dot-product attention","limit":5}' \
| jq '{retrieval_mode, vector_count, keyword_count, unique_count, combined_count}'
```

## Notes

- The PDF corpus under `tests/` is gitignored. Only the evaluation scripts are tracked.
- `jq` makes the curl output readable: `brew install jq`.
