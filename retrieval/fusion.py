"""
Reciprocal Rank Fusion (RRF) for combining Weaviate and Elasticsearch search results.
"""


def combine_search_results_with_rrf(
    weaviate_results: dict,
    elasticsearch_results: dict,
    search_limit: int,
    rrf_k: int = 60,
):
    """
    Combine Weaviate and Elasticsearch search results using Reciprocal Rank Fusion (RRF) algorithm.

    Args:
        weaviate_results: Results from Weaviate search
        elasticsearch_results: Results from Elasticsearch search
        search_limit: Maximum number of results to return
        rrf_k: RRF constant (default 60)

    Returns:
        Tuple of (final_results, combined_results, weaviate_count, elasticsearch_count)
    """
    # Track ranks for each result in each source
    # Key: (file_id, chunk_index), Value: dict with result data and ranks
    combined_results = {}

    # Process Weaviate results with ranks (1-indexed)
    for rank, obj in enumerate(weaviate_results.get("objects", []), start=1):
        key = (obj.get("file_id"), obj.get("chunk_index"))
        if key not in combined_results:
            combined_results[key] = {
                **obj,
                "sources": ["weaviate"],
                "weaviate_rank": rank,
                "elasticsearch_rank": None,
            }
        else:
            # Update existing result with Weaviate rank
            combined_results[key]["weaviate_rank"] = rank
            if "weaviate" not in combined_results[key]["sources"]:
                combined_results[key]["sources"].append("weaviate")
            # Preserve Weaviate-specific metadata
            if "distance" in obj:
                combined_results[key]["distance"] = obj["distance"]

    # Process Elasticsearch results with ranks (1-indexed)
    for rank, obj in enumerate(elasticsearch_results.get("objects", []), start=1):
        key = (obj.get("file_id"), obj.get("chunk_index"))
        if key not in combined_results:
            combined_results[key] = {
                **obj,
                "sources": ["elasticsearch"],
                "weaviate_rank": None,
                "elasticsearch_rank": rank,
            }
        else:
            # Update existing result with Elasticsearch rank
            combined_results[key]["elasticsearch_rank"] = rank
            if "elasticsearch" not in combined_results[key]["sources"]:
                combined_results[key]["sources"].append("elasticsearch")
            # Preserve Elasticsearch-specific metadata
            if "score" in obj:
                combined_results[key]["score"] = obj["score"]
            if "highlight" in obj:
                combined_results[key]["highlight"] = obj["highlight"]

    # Calculate RRF scores for each result
    # RRF_score = sum(1 / (k + rank)) for each source where result appears
    for key, result in combined_results.items():
        rrf_score = 0.0

        # Add contribution from Weaviate rank if present
        if result.get("weaviate_rank") is not None:
            rrf_score += 1.0 / (rrf_k + result["weaviate_rank"])

        # Add contribution from Elasticsearch rank if present
        if result.get("elasticsearch_rank") is not None:
            rrf_score += 1.0 / (rrf_k + result["elasticsearch_rank"])

        result["rrf_score"] = rrf_score

    # Convert to list and sort by RRF score (descending)
    final_results = list(combined_results.values())
    final_results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

    # Limit to requested number of results
    final_results = final_results[:search_limit]

    weaviate_count = len(weaviate_results.get("objects", []))
    elasticsearch_count = len(elasticsearch_results.get("objects", []))

    return final_results, combined_results, weaviate_count, elasticsearch_count
