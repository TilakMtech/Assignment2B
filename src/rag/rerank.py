"""Cross-encoder reranking of the top-10 candidates of the best Part B method."""
import time

import torch
from sentence_transformers import CrossEncoder

from src.rag import config


def load_reranker(name=config.RERANKER):
    return CrossEncoder(name, max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")


def rerank(reranker, query, candidates, chunks, keep=3):
    """Re-scores (query, chunk) pairs with the cross-encoder and keeps the best `keep`."""
    scores = reranker.predict([(query, chunks[cid].text) for cid, _ in candidates], show_progress_bar=False)
    order = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [(cid, float(s)) for (cid, _), s in order[:keep]]


def timed_rerank(reranker, queries, candidates_per_query, chunks, keep=3):
    rerank(reranker, queries[0]["query"], candidates_per_query[0], chunks, keep)  # warm-up
    out, times = [], []
    for q, cands in zip(queries, candidates_per_query):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out.append(rerank(reranker, q["query"], cands, chunks, keep))
        times.append(1000 * (time.perf_counter() - t0))
    return out, times
