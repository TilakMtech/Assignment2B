"""Paths, model names and settings for the Assignment 2B RAG pipeline.

Every value can be overridden with an environment variable of the same name.
"""
import os
from pathlib import Path


def _env(name, default, cast=str):
    value = os.environ.get(name)
    return cast(value) if value not in (None, "") else default


ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = Path(_env("CORPUS_DIR", str(ROOT / "data" / "domain_corpus")))
QUERIES = ROOT / "data" / "queries.json"
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
for _p in (RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ---- retrieval models
# bge-small-en-v1.5: 384-dim, 512-token limit, so 100 / 400 / 500-token chunks all fit without
# truncation (all-MiniLM-L6-v2 would cut everything after 256 tokens).
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_MAX_TOKENS = _env("EMBED_MAX_TOKENS", 512, int)
EMBED_BATCH = _env("EMBED_BATCH", 64, int)
# bge v1.5 recommends this instruction on short retrieval queries (not on passages).
QUERY_INSTRUCTION = _env("QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: ")
RERANKER = _env("RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZES = [100, 400, 500]   # brief: 100 and 400 in the text, 100 and 500 in its table
CHUNK_SIZE = _env("CHUNK_SIZE", 0, int)  # 0 = use the size chosen in Part A
RRF_K = 60
RRF_DEPTH = 20
TOP_K = 5

# ---- generation / judging
# Assignment 1 model = CPT checkpoint + QLoRA Adapter B from the CorpPolicyLM project.
A1_PROJECT = Path(_env("A1_PROJECT", str(Path.home() / "datavol-1" / "CorpPolicyLM")))
A1_BASE = Path(_env("A1_BASE", str(A1_PROJECT / "models" / "cpt")))
A1_ADAPTER = Path(_env("A1_ADAPTER", str(A1_PROJECT / "models" / "adapters" / "adapter_B")))
# Pre-quantised 4-bit (bitsandbytes NF4) copies of the official instruct models: ~5.5 GB each
# instead of ~15 GB, so they fit the lab disk and GPU. Weights = the official models.
GENERATOR = _env("GENERATOR", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
GENERATOR_NAME = _env("GENERATOR_NAME", "Qwen2.5-7B-Instruct (4-bit NF4)")
JUDGE = _env("JUDGE", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
JUDGE_NAME = _env("JUDGE_NAME", "Llama-3.1-8B-Instruct (4-bit NF4)")
GEN_MAX_NEW_TOKENS = _env("GEN_MAX_NEW_TOKENS", 200, int)
JUDGE_MAX_NEW_TOKENS = _env("JUDGE_MAX_NEW_TOKENS", 200, int)
NOT_FOUND = "Not found in the documents"
