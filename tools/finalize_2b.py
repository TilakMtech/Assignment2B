"""Fills in the D2.2-D2.4 write-ups (written after reading the judge's verdicts of the lab run)
in the executed notebook and re-exports the HTML. No models, no kernel; < 1 minute.

    python tools/finalize_2b.py
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "Group4_Assignment2B_RAG.ipynb"
PENDING = "*PENDING — written after the lab run.*"

D22 = """I read the full answers, expected answers and chunks of three queries, Q2, Q5 and Q8. Each has two verdicts, six in total, and I agree with **3 of the 6**.

| Query | Judge: correct / supported | My verdict | Agree? |
|---|---|---|---|
| Q2 Astral vigil mechanism | partly correct / yes | partly correct / yes | 2 of 2 |
| Q5 Bajaj verbal complaint | correct / yes | **partly correct / no** | 0 of 2 |
| Q8 JioStar retaliation | partly correct / yes | **wrong** / yes | 1 of 2 |

* **Q2:** The answer gives the legal mandate (Companies Act, SEBI regulation 22) and the commitment to fair, transparent conduct, both stated in chunk [1]. It misses the expected main point that the mechanism helps detect and deter fraud, so *partly correct* and *supported* are right. The judge's REASON, however, names an "objective … to extract maximum information" that is not part of the expected answer: the right verdict for a partly wrong reason.
* **Q5:** The answer says the verbal complaint is converted "when it is received by the ethics and compliance team". Chunk [3] says it is converted *by the receiver of the complaint*, with the complainant's consent. The ethics and compliance team appears in a different clause (the reporting route in 6.3). So the answer is *partly correct*, and that statement is *not supported*.
* **Disagreement explained, Q8:** The generator answered "Not found in the documents", while the expected answer exists and the span is in chunk [3]. By the judge prompt's own definition this is **wrong**. The judge chose *partly correct*, and its REASON describes an answer that was never given ("states that retaliation for taking part in an investigation is included"). It graded the expected answer and the chunks instead of the system answer. *Supported = yes* is right, because a "Not found" reply makes no claims.

**What the spot-check shows about the 3B judge.** It never answered *no* to SUPPORTED: 10 of 10 answers were marked supported. Its explanations sometimes do not match the answer it was given. **Corrected counts** from my reading of all 10 answers:
* Correct 5 (Q1, Q4, Q7, Q9, Q10)
* Partly correct 2 (Q2, Q5)
* Wrong 3 (Q3, Q6, Q8)
* Supported 9 of 10 (Q5 is not)

The judge's counts were 6 / 2 / 2 / 10. A larger judge (Llama-3.1-8B, which `JUDGE=` selects when there is enough disk) would be the first improvement."""

D23 = """**Chosen failure: Q6 (wrong), and the cause is retrieval.** The answer span is clause 4.1.1, *"Responsibilities of Managers: All managers at BFSL must ensure that no employee is subject to harassment …"*. It is in none of the three chunks shown above:
* [1] is the policy's *Purpose & Background*;
* [2] is the sign-off sheet and table of contents, which matched words such as "Policy Guidelines" and "Responsibilities";
* [3] comes from a different organisation's manual (IIM Raipur).

The generator therefore had nothing to compare and correctly replied "Not found in the documents". The needed information was **not** in the chunks, so the error comes from retrieval, not generation.

In Part B, BM25 did not rank the managers' clause in its top 5. Dense search did, at rank 3, but BM25 was the method passed to the reranker, so the cross-encoder never saw that chunk.

**For contrast, Q3 and Q8 are generation failures.** The answer span *was* in the chunks ([2] for Q3, [3] for Q8), yet the 3B generator replied "Not found". In both cases the span was not in chunk [1]."""

D24 = """**Fix: rerank a pool that combines dense and BM25 results, instead of BM25's top 10 alone.** Concretely, pass the RRF-fused top 20 (or simply the union of dense top 10 and BM25 top 10) to the cross-encoder, and keep its top 3.

* **Why it fixes Q6:** dense search ranks the managers'-responsibilities chunk 3rd while BM25 misses it. In a combined pool the cross-encoder would score it against the question, and its wording ("Responsibilities of Managers … ensure … no employee is subject to harassment") matches the question far better than the table-of-contents chunk that BM25 preferred.
* **Why it costs little:** the two methods fail on different queries (Q6 is found only by dense, Q5 is ranked 1st only by BM25), so a combined pool loses nothing that either finds. Reranking 20 candidates instead of 10 adds roughly 20 ms per query, which is negligible next to generation.

For the generation failures (Q3, Q8), a separate fix would add one sentence to the prompt: *"Read all three chunks before deciding the answer is not there."* Alternatively, use the 7B generator."""


def main():
    nb = nbformat.read(NB, as_version=4)
    d22 = [c for c in nb.cells if c.cell_type == "markdown" and "### D2.2 Spot-check" in c.source]
    tail = [c for c in nb.cells if c.cell_type == "markdown" and "### D2.4 Suggested fix" in c.source]
    if len(d22) != 1 or len(tail) != 1:
        sys.exit("Could not find the D2.2 / D2.4 markdown cells.")
    if PENDING in d22[0].source:
        d22[0].source = d22[0].source.replace(PENDING, D22, 1)
    if PENDING in tail[0].source:
        tail[0].source = tail[0].source.replace(PENDING, D23, 1).replace(PENDING, D24, 1)
    left = sum(PENDING in c.source for c in nb.cells)
    nbformat.write(nb, NB)
    print(f"D2.2-D2.4 written; PENDING left: {left}")
    sys.path.insert(0, str(ROOT / "tools"))
    from run_notebook import export
    export()
    print("Exported", NB.with_suffix(".html"))


if __name__ == "__main__":
    main()
