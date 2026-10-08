"""Fills in the seven write-ups that depend on the lab run (A2.2, B4, C4, the D1 switch, D2.2, D2.3, D2.4)
in the executed notebook and re-exports the HTML. The texts were written from the run-3 outputs
(outputs/results/*.csv). No models, no kernel, CPU only; < 1 minute.

    python tools/finalize_2b.py
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "Group4_Assignment2B_RAG.ipynb"
PENDING = "*PENDING — written after the lab run.*"

A22 = """**Chosen: 400 tokens.** 400 and 500 tie on hit rate@3 (70%), and each keeps the complete answer span inside one chunk for 9 of the 10 queries. 400 sends fewer tokens to the generator (1,173 instead of 1,457 per query for the top 3). Its longest prompt, plus 200 answer tokens, is 1,764 Qwen tokens, far below Qwen's 32,768-token limit.
* **100 tokens** is too small. Only 3 of the 10 answer spans fit inside a single chunk, so hit rate@3 drops to 20%.
* **500 tokens** gains nothing over 400 and makes the prompts longer. Measured with the Assignment 1 tokenizer, 8 of the 10 prompts would exceed the A1 model's 2,048-token context once 200 answer tokens are reserved. With 400 tokens, 2 of 10 would still exceed it (max 2,081). This is one reason the A1 model is not the final generator (see D1)."""

B4 = """**BM25 works best:** hit rate@1 70%, @3 90%, @5 90%, against 50 / 70 / 80 for dense search and 60 / 80 / 80 for hybrid. It is also the fastest (1.96 ms per query).

The queries name specific organisations and use the policies' own terms, such as "verbal complaint", "JioStar" and "Aurobindo". Exact keyword matching rewards these terms. The small general-purpose embedding model instead finds chunks that are similar in topic, and many policies in this corpus share topics (for example, five different harassment policies).

**Example, Q5** ("How is a verbal sexual-harassment complaint converted for formal handling at Bajaj Broking?"): BM25 ranks the span chunk 2nd, because it contains the exact words "verbal complaint", "converted" and "written complaint". Dense search does not have it in its top 5; it returns other harassment-procedure chunks. Hybrid search does not have it in its top 5 either: the dense ranking pushes it below rank 5 in the fusion. Hybrid helps only when both methods rank the chunk fairly high."""

C4 = """**No, on these 10 queries reranking made the results worse.** Hit rate@1 fell from 70% to 60% and hit rate@3 from 90% to 80%. The first result changed for 3 of 10 queries (30%), and reranking cost 21.3 ms per query.
* **Q9 improved:** the span chunk moved from rank 2 to rank 1.
* **Q3 got worse at rank 1:** a talent-development chunk about course-fee support ("financial support", "75%") took first place, and the laptop-policy span chunk moved to rank 2. It is still in the top 3.
* **Q7 was lost:** BM25 ranked the span chunk 652 first, but the cross-encoder put two other JioStar chunks (646, 645) and a Bajaj chunk above it, so it fell out of the top 3.

The reranker (ms-marco-MiniLM-L-6-v2) was trained on web search questions. It rewards chunks that sound like an answer to the question in general, and it does not know that the organisation name matters. BM25's exact term matching was already the stronger signal for this corpus."""

D1 = """The Assignment 1 model (TinyLlama-1.1B with the CPT and Adapter B) is shown with the answers saved from its trial run. Its model files are no longer on the lab disk, so it could not be run again with the revised prompt. **That trial used 500-token chunks and prompt version 1** (before the sentence on chunk labels was added). Both are stated in the table above, and its prompts were rebuilt exactly to count tokens with its own tokenizer.

**Problems observed:**
1. **The context is too long for the model.** TinyLlama has a 2,048-token context. The prompts of Q2, Q5 and Q6 are 2,165–2,195 tokens, so they exceed the limit before any answer is written. These are exactly the answers that break down into repeated or meaningless text (Q2: "The mechanism… The purchase…"). Q4 (2,040 tokens, leaving 8 for the answer) repeats one sentence. Only Q8 leaves the 200 answer tokens. This is evidence that the overlong context contributed to the failures.
2. **It does not use the chunks even when the prompt fits.** It cites no chunk in any of the 11 answers. Q8, the one prompt that fits, gives a plausible answer, but it cites nothing and adds details ("retaliation against witnesses") that are not in the span.
3. **It invents an answer for the out-of-corpus query.** For Q11 it describes a pet-friendly office with a leash rule. No such policy exists in the corpus.

The model was trained to answer HR questions from its own memory, not to follow a context-and-citation instruction. It is also too small for that task. **Switch:** the generator is Qwen2.5-3B-Instruct (4-bit), which has a 32,768-token context and is instruction-tuned. A 7B/8B model was preferred but did not fit the lab disk (4.9 GB home volume). With 400-token chunks, the longest Qwen prompt is 1,764 tokens including 200 answer tokens."""

D22 = """I read the full answer, the expected answer and the three chunks for **Q4, Q5 and Q8**. Each has two verdicts (correct, supported), so six in total. **I agree with 3 of the 6.**

| Query | Judge: correct / supported | My verdict | Agree? |
|---|---|---|---|
| Q4 MyGov laptop eligibility | partly correct / yes | partly correct / yes | 2 of 2 |
| Q5 Bajaj verbal complaint | partly correct / yes | **correct** / yes | 1 of 2 |
| Q8 JioStar retaliation | partly correct / yes | **wrong** / **yes** | 0 of 2 (see below) |

* **Q4 (agree):** The answer gives who decides (department head or immediate supervisor) and the criteria (role, job requirements, technical needs), both from chunk [1]. It leaves out the second half of the expected answer, the two arrangements (BYOD allowance and organisation-owned laptop) that depend on the device category. So *partly correct* and *supported*.
* **Q5 (disagree on correctness):** Chunk [2] says: *"In case of a verbal complaint, the complaint will be converted to a written complaint by the receiver of the complaint and consent of the complainant will be obtained."* The answer states exactly this, which is the whole expected answer. It should be *correct*. The judge's REASON asks for a point that is not in the expected answer ("informed of their rights"). It also lists "the consent of the complainant will be obtained" as unsupported, although that sentence is in chunk [2].
* **Q8 (disagree):** The generator answered "Not found in the documents", but chunk [1] contains the span: *"e. Retaliation for having reported or threatened to report harassment, or for opposing unlawful harassment, or for participating in an investigation"*. By the judge prompt's own definition, a "Not found" reply to an answerable question is **wrong**. The judge's REASON describes an answer that was never given ("states that retaliation … is included"), so it graded the expected answer instead of the system answer. *Supported = yes* is acceptable, because a "Not found" reply makes no claim. I count the pair as 0 of 2 because the reason it gives for "yes" is the invented answer.

**What the spot-check shows about the 3B judge.** It gave *partly correct* to 9 of the 10 answers and *correct* to none, and its reasons often name points that are not in the expected answer. My reading of all 10 answers gives:
* correct 1 (Q5);
* partly correct 5 (Q1, Q2, Q4, Q9, Q10);
* wrong 4 (Q3, Q6, Q7, Q8);
* supported 8 of 10.

Q1 is not supported: it attributes Bajaj Broking's employee list (chunk [2]) to Niramai. Q6 is not supported: its "broader oversight" claim is not in the chunks. The judge's counts were 0 / 9 / 1 and 9 of 10 supported. A larger judge (Llama-3.1-8B, selectable with `JUDGE=` when there is enough disk) would be the first improvement."""

D23 = """**Chosen failure: Q7 (judged wrong). The cause is reranking.** The trace shows the span chunk 652 at **rank 1 in BM25's top 10**, but it is **not in the final 3** after reranking. The cross-encoder ranked JioStar chunk 646, a Bajaj Broking chunk (59) and JioStar chunk 645 above it.

The three chunks the generator received are printed above:
* [1] and [3] are JioStar's definitions, coverage and complaint sections;
* [2] is Bajaj Broking's list of verbal conduct.

None contains JioStar's own list *"verbal… derogatory jokes… visual… offensive pictures, posters, drawings, gestures"*. The generator therefore correctly replied "Not found in the documents". **The needed information was not in the chunks, so this is not a generation error:** candidate retrieval found it, and reranking dropped it.

**For contrast, the other two causes:**
* **Q6 is a candidate-retrieval failure.** The managers' and individuals' responsibilities clause is not in BM25's top 10, so the reranker never saw it. It is not in the top 5 of dense or hybrid search either (Part B table).
* **Q8 is a generation failure.** The span chunk 652 is at rank 1 of the final 3, and it contains the exact retaliation clause. Yet the generator replied "Not found". Q3 is also a generation failure: the span chunk was [2], but the generator compared a course-fee policy instead."""

D24 = """**Suggested fix for Q7: always keep BM25's first result among the 3 chunks sent to the generator, and use the reranker only to choose the other two.** This may help Q7 and similar queries, because BM25's first result was already correct for 7 of 10 queries (hit rate@1 70%). The reranker's job would then be to add context, not to overrule a strong exact-match result.

I checked the retrieval part with the rankings already saved in `c_rerank_comparison.csv` (BM25 top 3 and reranked top 3 for each query). Keeping BM25's first chunk and filling with the reranker's top results gives:

| Final 3 chunks | Hit rate@1 | Hit rate@3 | Q7 span in final 3 | Time per query (measured parts) |
|---|---|---|---|---|
| BM25 top 10 → rerank → top 3 (used above) | 60% | 80% | no | 1.96 + 21.3 ms |
| BM25 #1 + reranker's best 2 others | 70% | 90% | yes (rank 1) | 1.96 + 21.3 ms |
| BM25 top 3, no reranking | 70% | 90% | yes (rank 1) | 1.96 ms |

The time is the same as the current pipeline: it is the measured BM25 and reranking times from Parts B and C, since nothing extra is computed. This is only the retrieval part. The answer quality was not re-measured, and Q8 shows that a correct chunk [1] does not guarantee a correct answer. So the fix **may help**, but it does not guarantee a better answer. It does not fix Q6: that chunk is not in BM25's top 10, so it needs better candidate retrieval (for example, a stronger embedding model).

For the generation failures (Q3, Q8), a separate change would add a prompt sentence: *"Read all three chunks before deciding the answer is not there,"* or use a larger generator."""

POOL_OLD = "**Testing the retrieval part of the fix** (retrieval and reranking only, no generation)."
POOL_NEW = ("**A second retrieval fix that was tested and did not help: a larger candidate pool** (retrieval and reranking only, "
            "no generation). The table below shows that pooling dense and BM25 candidates gave the same hit rates "
            "(60% / 80%) and doubled the time (36.2 ms vs 18.3 ms per query, measured). Q6 and Q7 are still missing "
            "from the final 3: a larger pool does not help when the reranker itself ranks the span chunk too low.")


def fill(nb, marker, text, count=1):
    cells = [c for c in nb.cells if c.cell_type == "markdown" and marker in c.source and PENDING in c.source]
    if len(cells) != 1:
        sys.exit(f"Expected one markdown cell with '{marker}' and PENDING, found {len(cells)}.")
    cells[0].source = cells[0].source.replace(PENDING, text, count)
    return cells[0]


def main():
    nb = nbformat.read(NB, as_version=4)
    fill(nb, "### A2.2 Chosen chunk size", A22)
    fill(nb, "### B4. Which method works best", B4)
    fill(nb, "### C4. Did reranking help?", C4)
    fill(nb, "**Problem observed with the Assignment 1 model, and the switch.**", D1)
    fill(nb, "### D2.2 Spot-check of the judge", D22)
    tail = fill(nb, "### D2.4 Suggested fix", D23)          # first PENDING in this cell is D2.3
    tail.source = tail.source.replace(PENDING, D24, 1)       # second is D2.4
    if POOL_OLD in tail.source:
        tail.source = tail.source.replace(POOL_OLD, POOL_NEW, 1)
    left = sum(PENDING in c.source for c in nb.cells)
    if left:
        sys.exit(f"PENDING still in {left} cell(s); notebook not saved.")
    nbformat.write(nb, NB)
    print("A2.2, B4, C4, D1, D2.2, D2.3, D2.4 written; PENDING left: 0")
    sys.path.insert(0, str(ROOT / "tools"))
    from run_notebook import export
    export()
    print("Exported", NB.with_suffix(".html"))


if __name__ == "__main__":
    main()
