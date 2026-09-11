"""
Reciprocal Rank Fusion (RRF) over a vector result set and a keyword result set.

RRF ignores score magnitudes and works purely on ranks, which is what makes it
safe across two engines whose scores are not on a comparable scale.
"""


def combine_search_results_with_rrf(
    vector_results: dict,
    keyword_results: dict,
    fetch_limit: int,
    rrf_k: int = 60,
):
    """
    Fuse a vector result set and a keyword result set with Reciprocal Rank Fusion.

        score = sum over sources of 1 / (k + rank)

    Args:
        vector_results:  {"objects": [...]} from semantic search
        keyword_results: {"objects": [...]} from BM25 search
        fetch_limit:     how many fused candidates to keep. This is the
                         pre-rerank budget, not the caller's final result count.
        rrf_k:           RRF constant (default 60)

    Returns:
        (fused_results, combined_results, vector_count, keyword_count)
    """
    # Key: (file_id, chunk_index) -> result data plus its rank in each source
    combined_results = {}

    # Vector results, ranked 1-indexed
    for rank, obj in enumerate(vector_results.get("objects", []), start=1):
        key = (obj.get("file_id"), obj.get("chunk_index"))
        if key not in combined_results:
            combined_results[key] = {
                **obj,
                "sources": ["vector"],
                "vector_rank": rank,
                "keyword_rank": None,
            }
        else:
            combined_results[key]["vector_rank"] = rank
            if "vector" not in combined_results[key]["sources"]:
                combined_results[key]["sources"].append("vector")
            # Preserve the vector-specific metric
            if "distance" in obj:
                combined_results[key]["distance"] = obj["distance"]

    # Keyword results, ranked 1-indexed
    for rank, obj in enumerate(keyword_results.get("objects", []), start=1):
        key = (obj.get("file_id"), obj.get("chunk_index"))
        if key not in combined_results:
            combined_results[key] = {
                **obj,
                "sources": ["keyword"],
                "vector_rank": None,
                "keyword_rank": rank,
            }
        else:
            combined_results[key]["keyword_rank"] = rank
            if "keyword" not in combined_results[key]["sources"]:
                combined_results[key]["sources"].append("keyword")
            # Preserve the keyword-specific metric
            if "score" in obj:
                combined_results[key]["score"] = obj["score"]

    # RRF score: one contribution per source the result appears in
    for result in combined_results.values():
        rrf_score = 0.0
        if result.get("vector_rank") is not None:
            rrf_score += 1.0 / (rrf_k + result["vector_rank"])
        if result.get("keyword_rank") is not None:
            rrf_score += 1.0 / (rrf_k + result["keyword_rank"])
        result["rrf_score"] = rrf_score

    fused_results = list(combined_results.values())
    fused_results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    fused_results = fused_results[:fetch_limit]

    vector_count = len(vector_results.get("objects", []))
    keyword_count = len(keyword_results.get("objects", []))

    return fused_results, combined_results, vector_count, keyword_count
