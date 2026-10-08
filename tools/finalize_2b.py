"""Fills in the seven write-ups that depend on the lab run (A2.2, B4, C4, the D1 switch, D2.2, D2.3, D2.4)
in the executed notebook and re-exports the HTML. The texts were written from the run-3 outputs
(outputs/results/*.csv). Safe to run again on a notebook finalised earlier: each section is
replaced in place. No models, no kernel, CPU only; < 1 minute.

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

A likely reason, not tested separately: the queries name specific organisations and use the policies' own terms, such as "verbal complaint", "JioStar" and "Aurobindo". Exact keyword matching rewards these terms. The small general-purpose embedding model may instead favour chunks that are similar in topic, and many policies in this corpus share topics (for example, five different harassment policies).

**Example, Q5** ("How is a verbal sexual-harassment complaint converted for formal handling at Bajaj Broking?"): BM25 ranks the span chunk 2nd, because it contains the exact words "verbal complaint", "converted" and "written complaint". Dense search does not have it in its top 5; it returns other harassment-procedure chunks. In this run, RRF did not improve over BM25 and pushed Q5's relevant chunk below the top five."""

C4 = """**No, on these 10 queries reranking made the results worse.** Hit rate@1 fell from 70% to 60% and hit rate@3 from 90% to 80%. The first result changed for 3 of 10 queries (30%), and reranking cost 21.3 ms per query.
* **Q9 improved:** the span chunk moved from rank 2 to rank 1.
* **Q3 got worse at rank 1:** a talent-development chunk about course-fee support ("financial support", "75%") took first place, and the laptop-policy span chunk moved to rank 2. It is still in the top 3.
* **Q7 was lost:** BM25 ranked the span chunk 652 first, but the cross-encoder put two other JioStar chunks (646, 645) and a Bajaj chunk above it, so it fell out of the top 3.

These experiments do not show why. One possible explanation is that the reranker (ms-marco-MiniLM-L-6-v2) was trained on web search questions, so it may favour chunks that sound like a general answer and give little weight to the organisation name. In Q7, it ranked a Bajaj Broking chunk above the JioStar span chunk. In this run, BM25's exact term matching was the stronger ranking signal."""

D1 = """The Assignment 1 model (TinyLlama-1.1B with the CPT and Adapter B) is shown with the answers saved from its trial run. Its model files are no longer on the lab disk, so it could not be run again with the revised prompt. **That trial used 500-token chunks and prompt version 1** (before the sentence on chunk labels was added). Both are stated in the table above, and its prompts were rebuilt exactly to count tokens with its own tokenizer.

**Problems observed:**
1. **The prompts are too long for the model's context, a confounding factor.** TinyLlama has a 2,048-token context. The prompts of Q2, Q5 and Q6 are 2,165–2,195 tokens, so they exceed the limit before any answer is written. These are the three answers that break down into repeated or meaningless text (Q2: "The mechanism… The purchase…"). Q4 (2,040 tokens, leaving 8 for the answer) repeats one sentence. Only Q8 leaves the 200 answer tokens. The match is consistent with the overlong context contributing to the repetition, but it does not prove the cause: prompt length is confounded with the model and the prompt version. A rerun with prompts trimmed to fit the context would separate these, but **no such rerun was performed**, because the model files are no longer available.
2. **It does not use the chunks even when the prompt fits.** It cites no chunk in any of the 11 answers. Q8, the one prompt that fits, gives a plausible answer, but it cites nothing and adds details ("retaliation against witnesses") that are not in the span.
3. **It invents an answer for the out-of-corpus query.** For Q11 it describes a pet-friendly office with a leash rule. No such policy exists in the corpus.

Possible explanations, which these experiments do not test: the model was fine-tuned to answer HR questions from its own memory, not to follow a context-and-citation instruction; and a 1.1B model may have too little capacity for such instructions. Whatever the cause, it does not meet the requirements of this pipeline (use the chunks, cite them, say "Not found"). **Switch:** the generator is Qwen2.5-3B-Instruct (4-bit), which has a 32,768-token context and is instruction-tuned. A 7B/8B model was preferred but did not fit the lab disk (4.9 GB home volume). With 400-token chunks, the longest Qwen prompt is 1,764 tokens including 200 answer tokens."""

D22 = """I read the full answer, the expected answer and the three chunks for **Q4, Q5 and Q8**. Each has two labels (correct, supported), so six in total.

**My grading convention.** *Correct* means the answer gives what the question asks, accurately. Details that appear only in the reference answer, beyond what the question asks, are not required. *Partly correct* means it misses part of what the question asks, or contains a minor error. *Wrong* means it misses the main point, or says "Not found" when the answer is in the documents. The judge prompt instead compares against the full reference answer, which is stricter.

| Query | Judge: correct / supported | My verdict | Label agreement |
|---|---|---|---|
| Q4 MyGov laptop eligibility | partly correct / yes | **correct** / yes | 1 of 2 |
| Q5 Bajaj verbal complaint | partly correct / yes | **correct** / yes | 1 of 2 |
| Q8 JioStar retaliation | partly correct / yes | **wrong** / yes | 1 of 2 |

**Label agreement: 3 of 6.** Under a strict reference-completeness convention, Q4 would be partly correct, because it omits the BYOD and organisation-laptop arrangements in the reference answer. The judge would then agree on both Q4 labels, and the count would be 4 of 6.

* **Q4:** The question asks how eligibility is determined. The answer gives both the criteria (role, job requirements, technical needs) and who decides (department head or immediate supervisor), as stated in chunk [1]. It omits the two device arrangements, but the question does not ask for them, so I mark it *correct* and *supported*.
* **Q5:** Chunk [2] says: *"In case of a verbal complaint, the complaint will be converted to a written complaint by the receiver of the complaint and consent of the complainant will be obtained."* The answer states exactly this, so it is *correct*. The judge's REASON asks for a point that is in neither the question nor the reference answer ("informed of their rights"). It also lists "the consent of the complainant will be obtained" as unsupported, although that sentence is in chunk [2], which contradicts its own *supported = yes*.
* **Q8:** The generator answered "Not found in the documents", but chunk [1] contains the span: *"e. Retaliation for having reported or threatened to report harassment, or for opposing unlawful harassment, or for participating in an investigation"*. That makes the answer **wrong**, not partly correct. *Supported = yes* is acceptable, because a "Not found" reply makes no claim, so that label agrees. **Separately, the judge's explanation is faulty:** its REASON describes an answer that was never given ("states that retaliation … is included"). It appears to have graded the expected answer or the chunks, not the system answer.

**What the spot-check shows about the 3B judge.** It gave *partly correct* to 9 of the 10 answers and *correct* to none. Its reasons sometimes name points that are in neither the question nor the reference. Under my convention, all 10 answers give:
* correct 3: Q4, Q5, and Q9 (it gives the initial action the question asks for: a confidential preliminary enquiry involving the complainant within 3 days);
* partly correct 3: Q1 and Q2 each miss or misstate part of the answer; Q10 says the preliminary enquiry is conducted by Corporate HR, which the policy does not state, and omits its 3-day limit;
* wrong 4: Q3, Q6, Q7, Q8;
* supported 8 of 10.

Q1 is not supported: it attributes Bajaj Broking's employee list (chunk [2]) to Niramai. Q6 is not supported: its "broader oversight" claim is not in the chunks. The judge's counts were 0 / 9 / 1, and 9 of 10 supported. A larger judge (Llama-3.1-8B, selectable with `JUDGE=` when there is enough disk) would be the first improvement to try."""

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
            "from the final 3.")


def section(nb, start, end, text):
    """Replace everything between `start` and `end` (or the end of the cell) in the one markdown cell
    that contains `start`. Works on the PENDING template and on a notebook finalised earlier."""
    cells = [c for c in nb.cells if c.cell_type == "markdown" and start in c.source]
    if len(cells) != 1:
        sys.exit(f"Expected one markdown cell with '{start}', found {len(cells)}.")
    src = cells[0].source
    a = src.index(start) + len(start)
    a = src.index("\n", a) + 1 if start.startswith("###") else a       # keep the heading line
    b = min((src.index(e, a) for e in end if e in src[a:]), default=len(src))
    gap = " " if not start.startswith("###") else ""
    cells[0].source = src[:a] + gap + text + ("\n\n" if b < len(src) else "\n") + src[b:]
    return cells[0]


def main():
    nb = nbformat.read(NB, as_version=4)
    section(nb, "### A2.2 Chosen chunk size", [], A22)
    section(nb, "### B4. Which method works best", [], B4)
    section(nb, "### C4. Did reranking help?", [], C4)
    section(nb, "**Problem observed with the Assignment 1 model, and the switch.**", ["### D1.2"], D1)
    section(nb, "### D2.2 Spot-check of the judge", ["### D2.3"], D22)
    tail = [c for c in nb.cells if c.cell_type == "markdown" and "### D2.4 Suggested fix" in c.source]
    if len(tail) != 1:
        sys.exit("Could not find the D2.3/D2.4 markdown cell.")
    src = tail[0].source
    tail[0].source = "\n" + D23 + "\n\n" + src[src.index("### D2.4 Suggested fix"):]
    section(nb, "### D2.4 Suggested fix", [POOL_OLD, POOL_NEW[:40]], D24)
    src = tail[0].source                                       # pooled-test intro: template or earlier wording
    i = src.find(POOL_OLD) if POOL_OLD in src else src.find(POOL_NEW[:40])
    j = src.index(" The same cross-encoder reranks", i)
    tail[0].source = src[:i] + POOL_NEW + src[j:]
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
