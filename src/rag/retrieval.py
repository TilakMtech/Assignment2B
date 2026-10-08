"""Dense (FAISS), BM25 and hybrid (RRF) search, plus the hit-rate / latency evaluation."""
import re
import time

import faiss
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.rag import config
from src.rag.data import is_hit


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder(name=config.EMBED_MODEL, max_tokens=config.EMBED_MAX_TOKENS):
    model = SentenceTransformer(name, device=device())
    model.max_seq_length = max_tokens
    return model


class DenseIndex:
    """Embeds chunks, stores them in a FAISS IndexFlatIP and returns the top-k for a query.

    Embeddings are L2-normalised, so inner product = cosine similarity.
    """

    def __init__(self, chunks, embedder, batch_size=config.EMBED_BATCH,
                 query_instruction=config.QUERY_INSTRUCTION):
        self.chunks, self.embedder, self.query_instruction = chunks, embedder, query_instruction
        self.batch_size = batch_size
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        vectors = embedder.encode([c.text for c in chunks], batch_size=batch_size,
                                  normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.build_seconds = time.perf_counter() - t0
        self.dim = vectors.shape[1]

    def search(self, query, k=config.TOP_K):
        q = self.embedder.encode([self.query_instruction + query], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype("float32")
        scores, ids = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def report(self):
        return {"embedding model": config.EMBED_MODEL, "vector size": self.dim,
                "similarity": "cosine (inner product of L2-normalised vectors, FAISS IndexFlatIP)",
                "max input tokens": self.embedder.max_seq_length, "batch size": self.batch_size,
                "chunks indexed": len(self.chunks), "device": device(),
                "index build time (s)": round(self.build_seconds, 2)}


def bm25_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, chunks):
        self.chunks = chunks
        t0 = time.perf_counter()
        self.bm25 = BM25Okapi([bm25_tokens(c.text) for c in chunks])
        self.build_seconds = time.perf_counter() - t0

    def search(self, query, k=config.TOP_K):
        scores = self.bm25.get_scores(bm25_tokens(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


def rrf(dense_hits, bm25_hits, k=config.RRF_K, top=config.TOP_K):
    """Reciprocal Rank Fusion: score(d) = 1/(k + rank_dense) + 1/(k + rank_bm25), ranks start at 1;
    a chunk missing from one list gets nothing from that list."""
    scores = {}
    for hits in (dense_hits, bm25_hits):
        for rank, (cid, _) in enumerate(hits, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return fused[:top]


class HybridIndex:
    def __init__(self, dense, bm25, depth=config.RRF_DEPTH):
        self.dense, self.bm25, self.depth = dense, bm25, depth

    def search(self, query, k=config.TOP_K):
        return rrf(self.dense.search(query, self.depth), self.bm25.search(query, self.depth), top=k)


def timed_search(index, queries, k=config.TOP_K):
    """Runs every query once after one warm-up query; returns results and latency (ms) per query."""
    index.search(queries[0]["query"], k)  # warm-up (not timed)
    results, times = [], []
    for q in queries:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        hits = index.search(q["query"], k)
        times.append(1000 * (time.perf_counter() - t0))
        results.append(hits)
    return results, times


def hit_at(results, queries, chunks, k):
    """Share (%) of queries with the answer span in at least one of the top-k results."""
    return 100 * np.mean([any(is_hit(chunks[cid].text, q["answer_span"]) for cid, _ in hits[:k])
                          for hits, q in zip(results, queries)])


def first_hit_rank(hits, q, chunks):
    for rank, (cid, _) in enumerate(hits, start=1):
        if is_hit(chunks[cid].text, q["answer_span"]):
            return rank
    return None


def per_query_table(results, queries, chunks, k=3):
    rows = []
    for hits, q in zip(results, queries):
        rows.append({"qid": q["qid"], "query": q["query"],
                     "rank of first hit": first_hit_rank(hits, q, chunks),
                     f"top-{k} sources": ", ".join(f"[{cid}] {chunks[cid].source}" for cid, _ in hits[:k])})
    return pd.DataFrame(rows)
