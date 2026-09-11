"""
Shared hybrid retrieval pipeline.

Two strategies, chosen by whether an Elasticsearch store is available. They are
deliberately different algorithms, not one algorithm with a fallback:

    rrf              Weaviate vector search + Elasticsearch BM25, fused by
                     Reciprocal Rank Fusion. Two engines whose scores are not
                     comparable, so fusion works on ranks alone.

    weaviate_hybrid  Weaviate's built-in hybrid(), blending vector and BM25
                     internally by alpha in a single round trip. Cheaper, and
                     tunable on a continuous knob rather than by rank.

Both then optionally rerank with a cross-encoder. Because the two modes are not
interchangeable, every outcome reports which one ran, so an evaluation result
can say what it actually measured.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from retrieval.fusion import combine_search_results_with_rrf
from retrieval.hyde import generate_hypothetical_document
from retrieval.reranker import rerank_cross_encoder

logger = logging.getLogger(__name__)

RRF_K = 60
CANDIDATE_MULTIPLIER = 4


@dataclass
class HybridSearchOutcome:
    """Result of one hybrid retrieval, plus the diagnostics the endpoints report."""

    objects: List[dict]
    retrieval_mode: str          # "rrf" | "weaviate_hybrid"
    chunk_store: Any             # store to use for get_chunks_by_range
    stats: Dict[str, Any]        # response fields, merged into the JSON body


def run_hybrid_search(
    query: str,
    vector_store,
    keyword_store: Optional[Any] = None,
    *,
    limit: int = 5,
    use_hyde: bool = False,
    rerank: bool = True,
    alpha: float = 0.5,
) -> HybridSearchOutcome:
    """
    Run hybrid retrieval and return the top `limit` results.

    Args:
        query:         the user's query, used verbatim for keyword matching
                       and for reranking.
        vector_store:  WeaviateStore.
        keyword_store: ElasticsearchStore, or None to use Weaviate's own hybrid.
        use_hyde:      expand the query into a hypothetical passage and embed
                       that instead. Affects only the vector half.
        rerank:        apply cross-encoder reranking to the candidates.
        alpha:         vector/BM25 blend, used only in weaviate_hybrid mode.
    """
    fetch_limit = limit * CANDIDATE_MULTIPLIER

    # HyDE replaces only the text that gets embedded. Keyword matching and
    # reranking both stay on the user's actual words.
    vector_query = generate_hypothetical_document(query) if use_hyde else query
    if use_hyde:
        logger.info("HyDE expanded %r into a %d-char passage", query, len(vector_query))

    if keyword_store is not None:
        vector_results = vector_store.search(vector_query, n_results=fetch_limit)
        keyword_results = keyword_store.search(query, n_results=fetch_limit)
        candidates, combined, vector_count, keyword_count = combine_search_results_with_rrf(
            vector_results, keyword_results, fetch_limit, RRF_K
        )
        retrieval_mode = "rrf"
        stats: Dict[str, Any] = {
            "vector_count": vector_count,
            "keyword_count": keyword_count,
            "unique_count": len(combined),
            "rrf_k": RRF_K,
        }
    else:
        hybrid_results = vector_store.hybrid_search(
            query, vector_query, n_results=fetch_limit, alpha=alpha
        )
        candidates = hybrid_results["objects"]
        retrieval_mode = "weaviate_hybrid"
        stats = {
            "alpha": alpha,
            "candidate_count": len(candidates),
        }

    if rerank:
        objects = rerank_cross_encoder(query, candidates, top_n=limit)
    else:
        objects = candidates[:limit]

    return HybridSearchOutcome(
        objects=objects,
        retrieval_mode=retrieval_mode,
        # Chunk windows are fetched from whichever store holds the full text.
        chunk_store=keyword_store or vector_store,
        stats={
            "retrieval_mode": retrieval_mode,
            **stats,
            "combined_count": len(objects),
        },
    )
