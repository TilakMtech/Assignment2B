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


# (index in the A1 eval set, answer span copied from the source document)
SELECTION = [
    (0, "is applicable to all the employees appointed on the payroll of Niramai Health Analytix Private Limited"),
    (2, "Such vigil mechanism not only helps to detect fraud in organizations, but is also used as a corporate "
        "governance tool, which prevents and deters fraudulent activity"),
    (4, "provision of a fixed monthly disbursement (“Laptop Charges”) for use of personal laptops"),
    (5, "Eligibility will be determined by the department head or immediate supervisor"),
    (6, "In case of a verbal complaint, the complaint will be converted to a written complaint by the receiver "
        "of the complaint and consent of the complainant will be obtained"),
    (7, "Responsibilities of Managers: All managers at BFSL must ensure that no employee is subject to harassment"),
    (8, "b. Visual conduct such as derogatory and/or sexually oriented posters, offensive or obscene photography, "
        "cartoons, drawings or gestures"),
    (9, "Retaliation for having reported or threatened to report harassment, or for opposing unlawful "
        "harassment, or for participating in an investigation"),
    (10, "Preliminary enquiry will be conducted by involving the complainant in such a way that the matter will "
         "be dealt with utmost confidentiality within 3days time"),
    (11, "Corporate HR, Unit HR, concerned Department HOD hold detailed enquiry"),
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
    for n, (idx, span) in enumerate(SELECTION, 1):
        row = EVAL[idx]
        doc = (ROOT / "data/domain_corpus" / row["source"]).read_text(encoding="utf-8")
        assert norm(span) in norm(doc), f"span for eval #{idx} not found in {row['source']}"
        queries.append({"qid": f"Q{n}", "a1_eval_index": idx, "query": row["instruction"],
                        "expected_answer": row["response"], "question_type": row["question_type"],
                        "source": row["source"], "answer_span": span})
    corpus = " ".join(norm(p.read_text(encoding="utf-8")) for p in (ROOT / "data/domain_corpus").glob("*.txt"))
    assert not re.search(r"\bpets?\b|\bdogs?\b", corpus), "out-of-corpus query is answerable"
    out = {"queries": queries, "out_of_corpus": OUT_OF_CORPUS}
    (ROOT / "data/queries.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(queries)} queries + 1 out-of-corpus query to data/queries.json")


if __name__ == "__main__":
    main()
