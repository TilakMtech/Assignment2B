"""Builds data/queries.json: 10 queries from the Assignment 1 evaluation set
(data/a1_eval/instruction_eval.jsonl) with their expected answers and a marked
answer span - the sentence / short passage in the corpus that answers each one.
Every span is verified to occur in its source document (whitespace-normalised).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = [json.loads(l) for l in (ROOT / "data/a1_eval/instruction_eval.jsonl").open(encoding="utf-8")]


def norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


# (index in the A1 eval set, first words, last words) of the answer span. The span is the SHORTEST
# contiguous source passage that supports the complete expected answer - for comparison questions it
# covers both sides being compared. It is cut from the (whitespace-normalised) source document.
SELECTION = [
    (0, "is applicable to all the employees appointed", "Private Limited"),
    (2, "mandates that every Listed Company to establish a mechanism", "prevents and deters fraudulent activity"),
    (4, "(a) provision of a fixed monthly disbursement", "requiring advanced technical capabilities"),
    (5, "based on their role, job requirements, and technical needs", "head or immediate supervisor"),
    (6, "In case of a verbal complaint", "consent of the complainant will be obtained"),
    (7, "Responsibilities of Individual:", "are not victimized in any manner"),
    (8, "a. Verbal conduct such as epithets", "cartoons, drawings or gestures"),
    (9, "Retaliation for having reported or threatened to report harassment",
        "or for participating in an investigation"),
    (10, "Any complaint that is received through email address", "whether there is a prima facie case or not"),
    (11, "Preliminary enquiry will be conducted", "to the attention of Management amicably"),
]
# Left out: eval #1 (same passage and near-duplicate question as #0) and #3 (abstract
# reading of the same Astral sentence as #2).

OUT_OF_CORPUS = {
    "qid": "Q11",
    "query": "Does MyGov allow employees to bring their pet dogs to the office, and what rules apply?",
    "expected_answer": "Not found in the documents (no policy in the corpus covers pets at the office).",
}


def main():
    queries = []
    for n, (idx, first, last) in enumerate(SELECTION, 1):
        row = EVAL[idx]
        doc = re.sub(r"\s+", " ", (ROOT / "data/domain_corpus" / row["source"]).read_text(encoding="utf-8"))
        start = doc.find(first)
        assert start >= 0, f"start of span for eval #{idx} not found in {row['source']}"
        end = doc.find(last, start)
        assert end >= 0, f"end of span for eval #{idx} not found in {row['source']}"
        span = doc[start:end + len(last)]
        queries.append({"qid": f"Q{n}", "a1_eval_index": idx, "query": row["instruction"],
                        "expected_answer": row["response"], "question_type": row["question_type"],
                        "source": row["source"], "answer_span": span, "span_words": len(span.split())})
    corpus = " ".join(norm(p.read_text(encoding="utf-8")) for p in (ROOT / "data/domain_corpus").glob("*.txt"))
    assert not re.search(r"\bpets?\b|\bdogs?\b", corpus), "out-of-corpus query is answerable"
    out = {"queries": queries, "out_of_corpus": OUT_OF_CORPUS}
    (ROOT / "data/queries.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(queries)} queries + 1 out-of-corpus query to data/queries.json")


if __name__ == "__main__":
    main()
