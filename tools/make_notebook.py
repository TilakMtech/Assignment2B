"""Generates notebooks/Group4_Assignment2B_RAG.ipynb (one notebook for the whole assignment)."""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
cells = []
md = lambda s: cells.append(nbformat.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbformat.v4.new_code_cell(s.strip()))
PENDING = "*PENDING — written after the lab run.*"

md(r"""
# Assignment 2B — RAG Pipeline on the HR-policy corpus (Group 4)

A small, complete Retrieval-Augmented Generation system built on the Assignment 1 (CorpPolicyLM) domain corpus.

| Part | What is done | Marks |
|---|---|---|
| A | Dense search (FAISS) and choice of chunk size | 5 |
| B | Dense vs keyword (BM25) vs hybrid (RRF) search | 4 |
| C | Cross-encoder reranking | 4 |
| D | Grounded answer generation and an LLM judge | 7 |

**Reused from Assignment 1:**
* **Documents:** the 33 cleaned `.txt` policy documents (`data/domain_corpus/`, from 34 PDFs, 902 pages).
* **Queries:** 10 queries with expected answers from the Assignment 1 evaluation set (`data/a1_eval/instruction_eval.jsonl`).
* **Generator:** the fine-tuned Assignment 1 model (TinyLlama-1.1B CPT checkpoint + QLoRA Adapter B) is tried first in D1.

**Generator and judge used after the switch:** `Qwen2.5-3B-Instruct` generates and `Llama-3.2-3B-Instruct` judges, both pre-quantised to 4-bit. These are the 3B members of the two model families the brief names. The 7–8B versions (about 5.5 GB each) do not fit the lab server's 4.9 GB home disk, and the lab quota did not allow a server with a bigger volume. The code runs the 7–8B models unchanged when `GENERATOR` / `JUDGE` are set.

**Code:** the pipeline lives in `src/rag/`; this notebook calls it and shows every result:
* `data.py` — chunking and the hit test;
* `retrieval.py` — dense, BM25 and RRF search;
* `rerank.py` — cross-encoder reranking;
* `generation.py` — prompts, generator and judge.
""")

code(r"""
import os, sys, json, time, inspect, importlib.util
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
missing = [m for m in ("faiss", "rank_bm25", "sentence_transformers", "peft", "bitsandbytes") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing packages {missing}: in a terminal run `pip install -r requirements.txt`, then restart the kernel.")
import numpy as np, pandas as pd, torch, faiss, transformers, sentence_transformers
from IPython.display import display, Markdown
pd.set_option("display.max_colwidth", None)
from src.rag import config
from src.rag.data import load_corpus, load_queries, chunk_fixed, is_hit, answer_chunk_ids, norm
from src.rag import retrieval, rerank as rr, generation as gen
print("torch", torch.__version__, "| transformers", transformers.__version__, "| sentence-transformers",
      sentence_transformers.__version__, "| faiss", faiss.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
print(f"Hugging Face cache: {gen.hf_cache_dir()} ({gen.free_gb():.1f} GiB free)")
""")

md(r"""
## Before you start — queries, answer spans and the embedding model's limit
The 10 queries come from the Assignment 1 evaluation set (12 held-out pairs from 6 documents). Two were left out:
* #1 asks about the same Niramai passage as #0 with a near-identical question.
* #3 is an abstract re-reading of the same Astral sentence as #2.

For each query, the **answer span** is the **shortest contiguous source passage that supports the complete expected answer**.
* For comparison questions (Q3, Q6, Q7, Q10), the span covers **both** sides being compared. Q6, for example, runs from the individual employees' responsibilities through the managers' responsibilities.
* For Q2, Q4 and Q9 it also includes the condition or channel that the expected answer names.

Every span is cut from its source document (`tools/make_queries.py`), so it occurs verbatim. A retrieved chunk is a **hit** only if it contains the **whole** span; the comparison is case-insensitive and ignores line breaks. A chunk that holds only one side of a comparison is therefore not a hit.

*Revision note:* a first version used shorter spans, some covering only one side of a comparison. The stricter spans lower every hit rate below, and longer spans are more often split across a chunk boundary.

A separate 11th query, whose answer is not in the corpus, is used only in D1.
""")

code(r"""
docs = load_corpus()
queries, ooc = load_queries()
print(f"{len(docs)} documents, {sum(len(t.split()) for t in docs.values()):,} words")
display(pd.DataFrame(queries)[["qid", "query", "answer_span", "span_words", "source", "expected_answer"]])
print("Out-of-corpus query (D1.3):", ooc["query"])
""")

code(r"""
embedder = retrieval.load_embedder()
emb_tok = embedder.tokenizer
limit = embedder.max_seq_length
doc_tokens = {name: len(emb_tok(text, add_special_tokens=False, verbose=False)["input_ids"]) for name, text in docs.items()}
print(f"Embedding model: {config.EMBED_MODEL} | max input length: {limit} tokens "
      f"(incl. [CLS]/[SEP], so up to {limit - 2} content tokens per chunk)")
print(f"Corpus size with this tokenizer: {sum(doc_tokens.values()):,} tokens "
      f"(largest document {max(doc_tokens.values()):,}, smallest {min(doc_tokens.values()):,})")
""")

md(r"""
**Why not `all-MiniLM-L6-v2`?** Its limit is 256 tokens, so 400- and 500-token chunks would be silently cut to their first 256 tokens. We therefore use **`BAAI/bge-small-en-v1.5`** (384-dim, 512-token limit, the same size class as MiniLM), so every chunk size is embedded in full. Token counts are always taken with this model's own tokenizer. Following the model card, queries get the prefix *"Represent this sentence for searching relevant passages: "*; chunks are embedded as they are.

# Part A — Dense search and chunk size (5 marks)
## A1. Build dense search (2 marks)
### A1.1 One reusable function (1 mark)
`DenseIndex` embeds a set of chunks, builds a FAISS `IndexFlatIP` and returns the top-k chunks for a query.
""")

code(r"""
print(inspect.getsource(retrieval.DenseIndex))
""")

md(r"""
## A2. Choose a chunk size (3 marks)
### A2.1 Fixed-size chunking at 100, 400 and 500 tokens, no overlap (2 marks)
The brief's text asks for 100 and 400 tokens, while its table lists 100 and 500, so all three sizes are run. Chunks are cut every *n* tokens of the embedding tokenizer, never across documents, and their text is taken from the original document, so a span can be matched exactly.

**Columns:**
* *Total chunks* — the absolute count (the table's "(1M)" header appears to be a typo).
* *Queries answerable* — queries whose span lies inside a single chunk; a span that straddles a chunk boundary cannot be a hit at that size.
* *Prompt tokens* — the exact length of the fully formatted chat prompt (system message, rules, the three numbered chunks and the question), counted **with each generator's own tokenizer**, plus 200 tokens reserved for the answer. It is shown as the maximum over the 10 queries, together with how many queries exceed the model's context limit:
  * Assignment 1 model (TinyLlama): 2,048 tokens;
  * Qwen2.5: 32,768 tokens.
  These rows use the dense top 3 of each size. D1 repeats the check for the exact prompts that were sent.
""")

code(r"""
from transformers import AutoTokenizer
a1_tok = gen.a1_tokenizer()                              # Assignment 1 tokenizer + its Zephyr chat template
g_tok_count = AutoTokenizer.from_pretrained(config.GENERATOR)   # generator tokenizer only (a few MB)
G_CONTEXT = 32768                                        # Qwen2.5-3B/7B-Instruct context length
def budget(tok, size_chunks, hits):   # per-query prompt tokens (+ reserved answer tokens), top-3 chunks
    return [gen.chat_prompt_tokens(tok, gen.GEN_SYSTEM, gen.gen_prompt(q["query"], [size_chunks[c] for c, _ in h[:3]]))
            + config.GEN_MAX_NEW_TOKENS for q, h in zip(queries, hits)]
size_rows, size_runs = [], {}
for size in config.CHUNK_SIZES:
    assert size + 2 <= limit, f"{size}-token chunks exceed the embedding limit"
    chunks = chunk_fixed(docs, emb_tok, size)
    index = retrieval.DenseIndex(chunks, embedder)
    results, _ = retrieval.timed_search(index, queries, k=3)
    size_runs[size] = (chunks, index, results)
    size_rows.append({"Chunk size (tokens)": size, "Total chunks": len(chunks),
                      "Hit rate@3 (%)": retrieval.hit_at(results, queries, chunks, 3),
                      "Tokens sent per query (top 3)": round(np.mean([sum(chunks[c].n_tokens for c, _ in h[:3]) for h in results]), 1),
                      "Queries answerable (span inside one chunk)": sum(bool(answer_chunk_ids(chunks, q["answer_span"])) for q in queries),
                      "A1 prompt tokens, max (+200 answer)": max(a1_b := budget(a1_tok, chunks, results)),
                      "Queries over A1 limit (2,048)": sum(t > config.A1_CONTEXT for t in a1_b),
                      "Qwen prompt tokens, max (+200 answer)": max(g_b := budget(g_tok_count, chunks, results)),
                      "Queries over Qwen limit (32,768)": sum(t > G_CONTEXT for t in g_b),
                      "Index build time (s)": round(index.build_seconds, 2)})
size_table = pd.DataFrame(size_rows)
size_table.to_csv(config.RESULTS / "a2_chunk_sizes.csv", index=False)
display(size_table)
for size, (chunks, _, results) in size_runs.items():
    print(f"{size:>3} tokens - rank of first hit per query:",
          [retrieval.first_hit_rank(h, q, chunks) for h, q in zip(results, queries)])
""")

md(r"""
### Choosing the size (rule applied in code; `CHUNK_SIZE=<n>` overrides it)
1. Every prompt must fit the context of the generator that answers in D1 (Qwen2.5, 32,768 tokens), measured per query with its own tokenizer.
2. Among those sizes, the highest hit rate@3.
3. On a tie, fewer tokens sent per query.

The Assignment 1 model's 2,048-token limit is reported in the table rather than used as a filter. That model is only the first trial in D1; how its limit interacts with the chosen size is discussed in A2.2 and D1.
""")

code(r"""
def pick_size(table):
    ok = table[table["Queries over Qwen limit (32,768)"] == 0]
    best = ok.sort_values(["Hit rate@3 (%)", "Tokens sent per query (top 3)"], ascending=[False, True]).iloc[0]
    return int(best["Chunk size (tokens)"])
CHUNK_SIZE = config.CHUNK_SIZE or pick_size(size_table)
chunks, dense, _ = size_runs[CHUNK_SIZE]
print(f"Chosen chunk size: {CHUNK_SIZE} tokens ({len(chunks)} chunks)")
""")

md(r"""
### A1.2 Dense-search report for the final chunk set (1 mark)
Build time = embedding all chunks plus adding them to the FAISS index.
""")

code(r"""
a1_report = dense.report()
json.dump(a1_report, open(config.RESULTS / "a1_dense_report.json", "w"), indent=2)
display(pd.Series(a1_report).to_frame("value"))
""")

md(r"""
### A2.2 Chosen chunk size and why (1 mark)
""" + PENDING + r"""
""")

md(r"""
# Part B — Keyword and hybrid search (4 marks)
All three methods use the chunks chosen in Part A and return the top 5 for each of the same 10 queries.
* **Latency:** the time to return the top 5 for one query, averaged over the 10 queries, after one untimed warm-up query.
* **Hybrid:** fuses the **top 20** of each method with RRF(d) = 1/(60 + rank in dense list) + 1/(60 + rank in BM25 list). Ranks start at 1, and a chunk missing from one list gets nothing from that list.
""")

code(r"""
print(inspect.getsource(retrieval.rrf))
bm25 = retrieval.BM25Index(chunks)
hybrid = retrieval.HybridIndex(dense, bm25)
methods = {"Dense": dense, "BM25": bm25, "Hybrid (RRF)": hybrid}
runs, rows = {}, []
for name, index in methods.items():
    results, times = retrieval.timed_search(index, queries, k=5)
    runs[name] = results
    rows.append({"Method": name, "Average query latency (ms)": round(float(np.mean(times)), 2),
                 "Hit rate@1 (%)": retrieval.hit_at(results, queries, chunks, 1),
                 "Hit rate@3 (%)": retrieval.hit_at(results, queries, chunks, 3),
                 "Hit rate@5 (%)": retrieval.hit_at(results, queries, chunks, 5)})
search_table = pd.DataFrame(rows)
search_table.to_csv(config.RESULTS / "b_search_comparison.csv", index=False)
display(search_table)
ranks = pd.DataFrame({"qid": [q["qid"] for q in queries], "query": [q["query"] for q in queries],
                      **{f"{m}: rank of first hit": [retrieval.first_hit_rank(h, q, chunks) for h, q in zip(runs[m], queries)] for m in runs}})
display(ranks)
""")

md(r"""
### Best method (rule applied in code; `BEST_METHOD=<name>` overrides it)
The best method has the highest hit rate@3; ties are broken by hit rate@1, then by lower latency.
""")

code(r"""
best_row = search_table.sort_values(["Hit rate@3 (%)", "Hit rate@1 (%)", "Average query latency (ms)"],
                                    ascending=[False, False, True]).iloc[0]
BEST = os.environ.get("BEST_METHOD") or best_row["Method"]
print("Best method for Part C:", BEST)
""")

md(r"""
### B4. Which method works best and why, with one example query (1 mark)
""" + PENDING + r"""
""")

md(r"""
# Part C — Reranking (4 marks)
### C1. Rerank (1 mark)
The best Part B method returns the top 10 for each query. A cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2`, re-scores each (query, chunk) pair and the top 3 are kept. Reranking time per query covers scoring the 10 pairs and sorting them, averaged after one warm-up query.
""")

code(r"""
reranker = rr.load_reranker()
before10, _ = retrieval.timed_search(methods[BEST], queries, k=10)
after3, rerank_ms = rr.timed_rerank(reranker, queries, before10, chunks, keep=3)
""")

md(r"""
### C2. Top 3 before and after reranking for each query (1 mark)
`[chunk id] source`; ✓ marks a chunk that contains the answer span.
""")

code(r"""
def show(hits, q):
    return " | ".join(f"[{c}] {chunks[c].source}{' ✓' if is_hit(chunks[c].text, q['answer_span']) else ''}" for c, _ in hits[:3])
compare = pd.DataFrame([{"qid": q["qid"], "query": q["query"], f"top 3 before ({BEST})": show(b, q),
                         "top 3 after reranking": show(a, q), "first result changed": b[0][0] != a[0][0]}
                        for q, b, a in zip(queries, before10, after3)])
compare.to_csv(config.RESULTS / "c_rerank_comparison.csv", index=False)
display(compare)
""")

md(r"""
### C3. Reranking measures (1 mark)
*Rank change rate* is the share of the 10 queries whose first result changes after reranking.
""")

code(r"""
rerank_table = pd.DataFrame({"Measure": ["Reranking time per query (ms)", "Rank change rate (%)",
                                         f"Hit rate@1 before reranking (from Part B, {BEST})", "Hit rate@1 after reranking",
                                         f"Hit rate@3 before reranking ({BEST})", "Hit rate@3 after reranking"],
                             "Value": [round(float(np.mean(rerank_ms)), 1), 100 * compare["first result changed"].mean(),
                                       retrieval.hit_at(before10, queries, chunks, 1), retrieval.hit_at(after3, queries, chunks, 1),
                                       retrieval.hit_at(before10, queries, chunks, 3), retrieval.hit_at(after3, queries, chunks, 3)]})
rerank_table.to_csv(config.RESULTS / "c_rerank_measures.csv", index=False)
display(rerank_table)
""")

md(r"""
### C4. Did reranking help? (1 mark)
""" + PENDING + r"""
""")

md(r"""
# Part D — Generate and check answers (7 marks)
## D1. Generate grounded answers (3 marks)
### D1.1 The prompt (1 mark)
Each query gets the **top 3 chunks after reranking**, numbered [1]–[3]. Decoding is greedy (temperature 0).
""")

code(r"""
print("SYSTEM:", gen.GEN_SYSTEM, "\n")
print(gen.GEN_TEMPLATE)
""")

code(r"""
# retrieval for the out-of-corpus query uses the same pipeline (best method top 10 -> rerank -> top 3)
ooc_q = {"qid": ooc["qid"], "query": ooc["query"], "expected_answer": ooc["expected_answer"], "answer_span": ""}
ooc_after3 = rr.rerank(reranker, ooc_q["query"], methods[BEST].search(ooc_q["query"], 10), chunks, keep=3)
all_q, all_ctx = queries + [ooc_q], after3 + [ooc_after3]
contexts = [[chunks[c] for c, _ in hits] for hits in all_ctx]
prompts = [gen.gen_prompt(q["query"], ctx) for q, ctx in zip(all_q, contexts)]
del embedder, reranker; gen.free()
print(f"Prompt length for Q1 with {CHUNK_SIZE}-token chunks: {len(prompts[0].split())} words")
""")

md(r"""
### First try: the Assignment 1 model (TinyLlama-1.1B CPT + Adapter B)
The brief allows switching to an instruct model if the Assignment 1 model ignores the chunks or never says "Not found". The same prompts are run through it first, and three behaviours are checked automatically: whether it cites a chunk, whether it says "Not found", and whether the answer stops cleanly.
""")

code(r"""
trial = json.loads((config.A1_TRIAL_DIR / "a1_trial_contexts.json").read_text())
if (config.A1_BASE / "config.json").exists() and (config.A1_ADAPTER / "adapter_config.json").exists():
    a1_model, a1m_tok = gen.load_a1_model()
    a1_prompts, source_note = prompts, "generated in this run with the current prompt"
    a1_answers = pd.DataFrame([{"qid": q["qid"], "query": q["query"], "A1 model answer": gen.chat(a1_model, a1m_tok, gen.GEN_SYSTEM, p)}
                               for q, p in zip(all_q, a1_prompts)])
    del a1_model; gen.free()
else:   # the A1 model files were removed from the lab disk after its trial run: show the saved trial
    tchunks = size_runs[trial["chunk_size"]][0]
    for q in all_q:   # the saved chunk ids must point at the same documents in this run's chunking
        ids = trial["contexts"][q["qid"]]["chunk_ids"]
        assert all(c < len(tchunks) for c in ids) and [tchunks[c].source for c in ids] == trial["contexts"][q["qid"]]["sources"], \
            f"saved A1 trial contexts do not match this run's {trial['chunk_size']}-token chunks ({q['qid']})"
    a1_prompts = [gen.GEN_TEMPLATE_V1.format(chunks=gen.format_chunks([tchunks[c] for c in trial["contexts"][q["qid"]]["chunk_ids"]]),
                                             question=q["query"]) for q in all_q]
    a1_answers = pd.read_csv(config.A1_TRIAL_DIR / "d1_a1_model_answers.csv")[["qid", "query", "A1 model answer"]]
    source_note = (f"saved from its trial run ({trial['chunk_size']}-token chunks, BM25 top 10 -> reranked top 3, "
                   "prompt version 1 - before the citation-label sentence); the model files are no longer on the lab disk")
a1_answers["cites a chunk"] = [bool(gen.citation_check(a)[0]) for a in a1_answers["A1 model answer"]]
a1_answers["says Not found"] = [gen.says_not_found(a) for a in a1_answers["A1 model answer"]]
# exact prompt length with the A1 tokenizer and chat template, + 200 tokens reserved for the answer
a1_answers["A1 prompt tokens"] = [gen.chat_prompt_tokens(a1_tok, gen.GEN_SYSTEM, p) for p in a1_prompts]
a1_answers["prompt + 200 within 2,048"] = a1_answers["A1 prompt tokens"] + config.GEN_MAX_NEW_TOKENS <= config.A1_CONTEXT
a1_answers.to_csv(config.RESULTS / "d1_a1_model_answers.csv", index=False)
print("Assignment 1 model answers:", source_note)
display(a1_answers)
over = a1_answers[~a1_answers["prompt + 200 within 2,048"]]
print(f"A1 prompt tokens: max {a1_answers['A1 prompt tokens'].max()}, {len(over)}/11 prompts leave less than 200 tokens "
      f"for the answer within the 2,048-token context ({', '.join(over.qid) or 'none'})")
print(f"A1 model: cites a chunk in {a1_answers['cites a chunk'].sum()}/11 answers; "
      f"says 'Not found' for the out-of-corpus query: {bool(a1_answers['says Not found'].iloc[-1])}")
""")

md(r"""
**Problem observed with the Assignment 1 model, and the switch.** """ + PENDING + r"""

### D1.2 Answers of the generator (1 mark) and D1.3 the out-of-corpus query (1 mark)
""")

code(r"""
print(f"Generator: {config.GENERATOR_NAME}  [{config.GENERATOR}]  - free disk before download: {gen.free_gb():.1f} GiB")
gen.make_room(config.GENERATOR, drop=config.JUDGE)   # re-runs: the judge may still be cached
g_model, g_tok = gen.load_instruct(config.GENERATOR)
answers = [gen.chat(g_model, g_tok, gen.GEN_SYSTEM, p) for p in prompts]
del g_model; gen.free()
cit = [gen.citation_check(a, n_chunks=len(ctx)) for a, ctx in zip(answers, contexts)]
gen_table = pd.DataFrame([{"qid": q["qid"], "query": q["query"], "answer": a, "valid citations": v, "invalid citations": inv,
                           "prompt tokens (Qwen)": gen.chat_prompt_tokens(g_tok, gen.GEN_SYSTEM, p),
                           "chunks used (source)": " | ".join(f"[{i}] {c.source}" for i, c in enumerate(ctx, 1))}
                          for q, a, (v, inv), p, ctx in zip(all_q, answers, cit, prompts, contexts)])
gen_table.to_csv(config.RESULTS / "d1_generator_answers.csv", index=False)
json.dump([{"qid": q["qid"], "query": q["query"], "expected_answer": q["expected_answer"], "answer": a,
            "chunk_ids": [c for c, _ in h]} for q, a, h in zip(all_q, answers, all_ctx)],
          open(config.RESULTS / "d1_answers.json", "w"), indent=2, ensure_ascii=False)
display(gen_table.iloc[:10])
print("D1.3 - out-of-corpus query:", ooc_q["query"])
print("Answer:", answers[-1])
print("Says 'Not found in the documents':", gen.says_not_found(answers[-1]))
bad = gen_table[gen_table["invalid citations"].str.len() > 0]
uncited = gen_table[(gen_table["valid citations"].str.len() == 0) & ~gen_table.answer.map(gen.says_not_found)]
print(f"Citation check: {len(bad)} answers cite labels other than [1]-[3] ({', '.join(bad.qid) or 'none'}); "
      f"{len(uncited)} answers that are not 'Not found' cite no valid chunk ({', '.join(uncited.qid) or 'none'}). "
      "Answers are shown exactly as generated.")
print(f"Qwen prompt tokens: max {gen_table['prompt tokens (Qwen)'].max()} (+200 for the answer) of a 32,768-token context")
""")

md(r"""
## D2. Check the answers with an LLM judge (4 marks)
### D2.1 Judge prompt and results (1 mark)
**Judge model:** Llama (printed below), a different model family from the Qwen generator. Decoding is greedy, so it behaves like temperature 0 and gives the same verdict every time.

The judge is given the question, the expected answer from the Assignment 1 evaluation set, the three retrieved chunks and the generator's answer. It returns four fixed lines (`CORRECT`, `SUPPORTED`, `REASON`, `UNSUPPORTED`), which are parsed with code.

**Constrained decoding of the format.** On a first run, the 3B judge left the CORRECT and SUPPORTED fields empty in all 10 replies. The reply is therefore built line by line from the same prompt:
* **Labels:** for `CORRECT` and `SUPPORTED`, the judge chooses among the allowed labels only, taking the label whose first token it rates most probable. This is argmax, i.e. temperature 0.
* **Explanations:** `REASON` and `UNSUPPORTED` are then generated greedily, conditioned on the chosen labels.
""")

code(r"""
print("SYSTEM:", gen.JUDGE_SYSTEM, "\n")
print(gen.JUDGE_TEMPLATE)
""")

code(r"""
gen.make_room(config.JUDGE, drop=config.GENERATOR)   # small disk: one model at a time (answers are already saved)
print(f"Judge: {config.JUDGE_NAME}  [{config.JUDGE}]")
j_model, j_tok = gen.load_instruct(config.JUDGE)
judged = [gen.judge(j_model, j_tok, gen.judge_prompt(q["query"], q["expected_answer"], ctx, a))
          for q, a, ctx in zip(queries, answers[:10], contexts[:10])]
del j_model; gen.free()
verdicts = pd.DataFrame([{"qid": q["qid"], "query": q["query"], **v} for q, (v, _) in zip(queries, judged)])
raw = [r for _, r in judged]
verdicts["raw judge output"] = raw
verdicts.to_csv(config.RESULTS / "d2_judge_verdicts.csv", index=False)
display(verdicts[["qid", "query", "correct", "supported", "reason", "unsupported quote"]])
summary = pd.DataFrame({"Check": ["Correct", "Partly correct", "Wrong", "Supported by the chunks"],
                        "Count out of 10": [int((verdicts.correct == "correct").sum()), int((verdicts.correct == "partly correct").sum()),
                                            int((verdicts.correct == "wrong").sum()), int((verdicts.supported == "yes").sum())]})
display(summary)
print("Unparsed verdicts:", int((verdicts[["correct", "supported"]] == "unparsed").any(axis=1).sum()))
""")

md(r"""
### D2.2 Spot-check of the judge (1 mark)
""" + PENDING + r"""

### D2.3 Find the cause (1 mark)
For every query, the table below traces where the complete answer span was lost. There are three possible causes:
* **Candidate retrieval:** the span is not in the best method's top 10, so the reranker never saw it.
* **Reranking:** the span is in the top 10, but not in the final 3 chunks.
* **Generation:** the span is in the 3 chunks the generator received, so the generator caused the error.

After the table come the three chunks of every answer the judge marked wrong, partly correct or unsupported.
""")

code(r"""
def stage(q, b10, a3):
    r10, r3 = retrieval.first_hit_rank(b10, q, chunks), retrieval.first_hit_rank(a3, q, chunks)
    cause = ("candidate retrieval (not in top 10)" if r10 is None else
             "reranking (in top 10, not in final 3)" if r3 is None else "chunks OK - any error is generation")
    return r10, r3, cause
trace = pd.DataFrame([{"qid": q["qid"], f"rank in {BEST} top 10": r10, "rank in final 3": r3, "where the span was lost": cause,
                       "judge": f"{v.correct} / supported={v.supported}"}
                      for q, b, a, (_, v) in zip(queries, before10, after3, verdicts.iterrows())
                      for r10, r3, cause in [stage(q, b, a)]])
trace.to_csv(config.RESULTS / "d2_failure_trace.csv", index=False)
display(trace)
flagged = verdicts[(verdicts.correct != "correct") | (verdicts.supported != "yes")]
for _, v in flagged.iterrows():
    i = next(k for k, q in enumerate(queries) if q["qid"] == v.qid)
    display(Markdown(f"**{v.qid} — {queries[i]['query']}**  \nJudge: {v.correct}, supported = {v.supported}  \n"
                     f"Answer: {answers[i]}  \nExpected: {queries[i]['expected_answer']}  \n"
                     f"Where the span was lost: **{trace.loc[i, 'where the span was lost']}**"))
    for n, c in enumerate(contexts[i], 1):
        print(f"[{n}] ({c.source}) {' '.join(c.text.split())}\n")
if flagged.empty:
    print("No answer was marked wrong, partly correct or unsupported.")
""")

md(r"""
""" + PENDING + r"""

### D2.4 Suggested fix (1 mark)
""" + PENDING + r"""

**Testing the retrieval part of the fix** (retrieval and reranking only, no generation). The same cross-encoder reranks a pool that combines the dense top 10 and the BM25 top 10, and keeps the top 3. This is compared with the pipeline used above (BM25 top 10 → top 3).
""")

code(r"""
reranker = rr.load_reranker()
def pooled(q):
    pool = list(dict.fromkeys([c for c, _ in dense.search(q["query"], 10)] + [c for c, _ in bm25.search(q["query"], 10)]))
    return rr.rerank(reranker, q["query"], [(c, 0.0) for c in pool], chunks, keep=3), len(pool)
pooled(queries[0])                                             # warm-up
pool_res, pool_ms, pool_sizes = [], [], []
for q in queries:
    t0 = time.perf_counter(); res, n = pooled(q); pool_ms.append(1000 * (time.perf_counter() - t0))
    pool_res.append(res); pool_sizes.append(n)
cur_ms = []
for q in queries:
    t0 = time.perf_counter(); rr.rerank(reranker, q["query"], methods[BEST].search(q["query"], 10), chunks, keep=3)
    cur_ms.append(1000 * (time.perf_counter() - t0))
fix_table = pd.DataFrame({"Pipeline": [f"{BEST} top 10 -> rerank -> top 3 (used above)", "dense top 10 + BM25 top 10 -> rerank -> top 3"],
                          "Hit rate@1 (%)": [retrieval.hit_at(after3, queries, chunks, 1), retrieval.hit_at(pool_res, queries, chunks, 1)],
                          "Hit rate@3 (%)": [retrieval.hit_at(after3, queries, chunks, 3), retrieval.hit_at(pool_res, queries, chunks, 3)],
                          "Candidates reranked (avg)": [10, float(np.mean(pool_sizes))],
                          "Retrieval + rerank time per query (ms)": [round(float(np.mean(cur_ms)), 1), round(float(np.mean(pool_ms)), 1)]})
fix_table.to_csv(config.RESULTS / "d2_fix_test.csv", index=False)
display(fix_table)
display(pd.DataFrame({"qid": [q["qid"] for q in queries],
                      "rank in final 3 (used above)": [retrieval.first_hit_rank(a, q, chunks) for a, q in zip(after3, queries)],
                      "rank in final 3 (pooled)": [retrieval.first_hit_rank(p, q, chunks) for p, q in zip(pool_res, queries)]}))
""")

md(r"""
## Export
`nbconvert` exports the saved `.ipynb`, so save the notebook first (Ctrl+S), or run `python tools/export_html.py` after closing it.
""")

code(r"""
import subprocess
env = {**os.environ, "JUPYTER_PATH": "/opt/conda/share/jupyter"}
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "notebooks/Group4_Assignment2B_RAG.ipynb",
                "--output-dir", "notebooks"], check=False, env=env)
""")

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python_LLM", "display_name": "python_LLM", "language": "python"}
out = ROOT / "notebooks" / "Group4_Assignment2B_RAG.ipynb"
nbformat.write(nb, out)
print("Wrote", out, len(cells), "cells")
