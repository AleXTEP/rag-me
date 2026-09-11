"""
Retrieval quality evaluator.

Reads a golden dataset (JSONL) and measures Recall@K, MRR, and Precision@1
for each search mode: vector, keywords, hybrid (with/without HyDE, with/without reranker).

Usage:
    python tests/evaluate_retrieval.py [options]

Options:
    --dataset PATH    Golden dataset JSONL (default: tests/golden_dataset.jsonl)
    --api-url URL     API base URL (default: http://localhost:8000)
    --k N             Recall@K cutoff (default: 5)
    --modes           Comma-separated modes to test (default: all)
                      Choices: vector, keywords, hybrid, hybrid_hyde
    --output PATH     Save results as JSON (optional)

A "hit" at rank k means the expected chunk appears in the top-k results,
identified by matching file_id AND chunk_index.
"""

import argparse
import json
import sys
import time
from typing import Callable

import requests


# ── Match logic ──────────────────────────────────────────────────────────────

def is_hit(result: dict, expected_file_id: str, expected_chunk_index: int) -> bool:
    """True if this result is the expected chunk (same doc + adjacent chunk acceptable)."""
    return (
        result.get("file_id") == expected_file_id
        and result.get("chunk_index") == expected_chunk_index
    )


def find_rank(results: list[dict], file_id: str, chunk_index: int) -> int | None:
    """Return 1-based rank of the expected chunk, or None if not found."""
    for i, r in enumerate(results, 1):
        if is_hit(r, file_id, chunk_index):
            return i
    return None


# ── Search adapters ──────────────────────────────────────────────────────────

def search_vector(api_url: str, question: str, k: int) -> list[dict]:
    resp = requests.post(f"{api_url}/search/vector", json={"q": question, "limit": k}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("objects", [])


def search_keywords(api_url: str, question: str, k: int) -> list[dict]:
    resp = requests.post(f"{api_url}/search/keywords", json={"q": question, "limit": k}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("objects", [])


def search_hybrid(api_url: str, question: str, k: int, use_hyde: bool = False) -> list[dict]:
    resp = requests.post(
        f"{api_url}/search/double",
        json={"q": question, "limit": k, "use_hyde": use_hyde},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("objects", [])


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(ranks: list[int | None], doc_ranks: list[int | None], k: int) -> dict:
    n = len(ranks)
    hits = [r for r in ranks if r is not None and r <= k]
    doc_hits_at_2 = [r for r in doc_ranks if r is not None and r <= 2]

    recall_at_k = len(hits) / n if n else 0.0
    mrr = sum(1.0 / r for r in hits) / n if n else 0.0
    precision_at_1 = sum(1 for r in ranks if r == 1) / n if n else 0.0
    doc_recall_at_2 = len(doc_hits_at_2) / n if n else 0.0

    return {
        "recall_at_k":    round(recall_at_k, 4),
        "mrr":            round(mrr, 4),
        "precision_at_1": round(precision_at_1, 4),
        "doc_recall_at_2": round(doc_recall_at_2, 4),
        "hits":           len(hits),
        "doc_hits_at_2":  len(doc_hits_at_2),
        "total":          n,
        "k":              k,
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def find_doc_rank(results: list[dict], file_id: str) -> int | None:
    """Return 1-based rank of the first result matching the expected document."""
    for i, r in enumerate(results, 1):
        if r.get("file_id") == file_id:
            return i
    return None


def run_mode(
    name: str,
    pairs: list[dict],
    search_fn: Callable[[str, int], list[dict]],
    k: int,
) -> dict:
    ranks: list[int | None] = []
    doc_ranks: list[int | None] = []
    latencies: list[float] = []
    errors = 0

    print(f"\n  Running [{name}] ...")
    for i, pair in enumerate(pairs, 1):
        question     = pair["question"]
        file_id      = pair["file_id"]
        chunk_index  = pair["chunk_index"]

        t0 = time.perf_counter()
        try:
            results = search_fn(question, k)
            latency = time.perf_counter() - t0
        except Exception as e:
            print(f"    [{i}/{len(pairs)}] ERROR: {e}")
            errors += 1
            ranks.append(None)
            doc_ranks.append(None)
            latencies.append(0.0)
            continue

        rank = find_rank(results, file_id, chunk_index)
        doc_rank = find_doc_rank(results, file_id)
        ranks.append(rank)
        doc_ranks.append(doc_rank)
        latencies.append(latency)

        hit_marker = f"rank={rank}" if rank else "MISS"
        doc_marker = f"doc@{doc_rank}" if doc_rank and doc_rank <= 2 else ("doc>2" if doc_rank else "doc:MISS")
        print(f"    [{i}/{len(pairs)}] {hit_marker:10s} {doc_marker:10s}  ({latency:.2f}s)  {question[:55]}")

    metrics = compute_metrics(ranks, doc_ranks, k)
    metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 3) if latencies else 0
    metrics["errors"] = errors
    metrics["mode"] = name
    return metrics


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(all_metrics: list[dict], k: int):
    print("\n" + "=" * 82)
    print(f"{'Mode':<20} {'Recall@'+str(k):<12} {'DocRecall@2':<14} {'MRR':<10} {'P@1':<10} {'Avg ms':<10} {'Hits'}")
    print("-" * 82)
    for m in all_metrics:
        print(
            f"{m['mode']:<20} "
            f"{m['recall_at_k']:<12.3f} "
            f"{m['doc_recall_at_2']:<14.3f} "
            f"{m['mrr']:<10.3f} "
            f"{m['precision_at_1']:<10.3f} "
            f"{int(m['avg_latency_s']*1000):<10} "
            f"{m['hits']}/{m['total']}  (doc@2: {m['doc_hits_at_2']}/{m['total']})"
        )
    print("=" * 82)


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_MODES = ["vector", "keywords", "hybrid", "hybrid_hyde"]

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a golden dataset.")
    parser.add_argument("--dataset", default="tests/golden_dataset.jsonl")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--k",       type=int, default=5)
    parser.add_argument("--modes",   default=",".join(ALL_MODES))
    parser.add_argument("--output",  default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    # Load dataset
    try:
        with open(args.dataset, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Dataset not found: {args.dataset}")
        print("Run tests/generate_golden_dataset.py first.")
        sys.exit(1)

    print(f"Loaded {len(pairs)} Q&A pairs from {args.dataset}")
    print(f"API:    {args.api_url}")
    print(f"K:      {args.k}")

    # Health check
    try:
        requests.get(f"{args.api_url}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"\nCannot reach API at {args.api_url}: {e}")
        sys.exit(1)

    modes = [m.strip() for m in args.modes.split(",")]
    api  = args.api_url
    k    = args.k

    mode_map = {
        "vector":      lambda q, k: search_vector(api, q, k),
        "keywords":    lambda q, k: search_keywords(api, q, k),
        "hybrid":      lambda q, k: search_hybrid(api, q, k, use_hyde=False),
        "hybrid_hyde": lambda q, k: search_hybrid(api, q, k, use_hyde=True),
    }

    all_metrics = []
    for mode in modes:
        if mode not in mode_map:
            print(f"Unknown mode: {mode}. Choices: {', '.join(ALL_MODES)}")
            continue
        metrics = run_mode(mode, pairs, mode_map[mode], k)
        all_metrics.append(metrics)

    print_report(all_metrics, k)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
