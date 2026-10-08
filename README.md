# Group 4 — Assignment 2B: RAG Pipeline (HR-policy corpus)

One notebook (`notebooks/Group4_Assignment2B_RAG.ipynb`) covering:

* **Part A:** FAISS dense search and the choice of chunk size.
* **Part B:** dense vs BM25 vs hybrid (RRF) search.
* **Part C:** cross-encoder reranking.
* **Part D:** grounded generation and an LLM judge.

## Reused from Assignment 1 (CorpPolicyLM)
* `data/domain_corpus/*.txt`: the 33 cleaned documents (34 PDFs, 902 pages).
* `data/a1_eval/instruction_eval.jsonl`: the Assignment 1 evaluation set. `tools/make_queries.py` picks 10 queries from it, marks the answer span of each in the corpus, and writes `data/queries.json`.
* The Assignment 1 model (CPT checkpoint + Adapter B), read from the CorpPolicyLM project (`A1_PROJECT`).

## Models
| Role | Model |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim, 512-token limit) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | Assignment 1 model first; then `Qwen2.5-7B-Instruct`, from the pre-quantised 4-bit copy `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Judge | `Llama-3.1-8B-Instruct`, from the pre-quantised 4-bit copy `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |

## Lab run (GPU server, `python_LLM` venv)

### Server
* Mount `datavol-1`, which holds the venv and the CorpPolicyLM project with the A1 model.
* Mount a new 15 GB data volume (e.g. `datavol-2`) for this project and the model cache. The two 4-bit 7–8B models need about 11 GB.

### Commands
```bash
source ~/datavol-1/venv/bin/activate && export PYTHONNOUSERSITE=1
cd ~/datavol-2/Group4_Assignment2B
pip install -r requirements.txt
export HF_HOME=~/datavol-2/hf_cache A1_PROJECT=~/datavol-1/CorpPolicyLM
KERNEL=python_llm python -u tools/run_notebook.py      # runs every cell, saves, exports HTML
```

Settings can be overridden with environment variables (see `src/rag/config.py`), for example `CHUNK_SIZE=400` or `BEST_METHOD=BM25`.
