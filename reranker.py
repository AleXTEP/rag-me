from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL

_cross_encoder = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def rerank_cross_encoder(query: str, results: list, top_n: int) -> list:
    """
    Rerank results using a cross-encoder model.

    Args:
        query: The search query
        results: List of result dicts, each must have a "text" key
        top_n: Number of top results to return

    Returns:
        Top N results sorted by rerank_score (descending), each with "rerank_score" added
    """
    if not results:
        return []

    model = _get_cross_encoder()
    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_n]
