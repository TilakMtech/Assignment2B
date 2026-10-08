"""Grounded answer generation (Part D1) and the LLM judge (Part D2)."""
import gc
import os
import re
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.rag import config

# ----------------------------------------------------------------------------- prompts
GEN_SYSTEM = ("You are an HR-policy assistant. You answer questions strictly from the document chunks "
              "you are given and never use outside knowledge.")

GEN_TEMPLATE = """Answer the question using ONLY the numbered chunks below.

Rules:
1. Use only facts stated in the chunks. Do not add facts from outside knowledge.
2. Cite the chunk number(s) you used in square brackets after each sentence, for example [2].
3. If the chunks do not contain the answer, reply exactly: Not found in the documents.
4. Answer in 1 to 3 sentences.

Chunks:
{chunks}

Question: {question}
Answer:"""

JUDGE_SYSTEM = "You are a strict examiner who grades answers from an HR-policy question-answering system."

JUDGE_TEMPLATE = """Grade the SYSTEM ANSWER below. Check two things separately.

CHECK 1 - CORRECT: compare the system answer with the EXPECTED ANSWER.
- correct: it states the main point(s) of the expected answer and contradicts nothing in it.
- partly correct: it states some of the expected answer's points but misses an important one or adds an error, without contradicting the main point.
- wrong: it misses or contradicts the main point, answers a different question, or is "Not found in the documents" when the expected answer exists.
If the expected answer is itself "Not found in the documents", then a "Not found" reply is correct and any invented answer is wrong.
Ignore citation numbers and wording; judge the meaning.

CHECK 2 - SUPPORTED: compare every statement in the system answer with the RETRIEVED CHUNKS only.
- yes: every statement can be found in (or directly follows from) the chunks.
- no: at least one statement is not in the chunks.
A "Not found in the documents" reply makes no claims, so it counts as supported (yes).

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

RETRIEVED CHUNKS:
{chunks}

SYSTEM ANSWER:
{answer}

Reply with exactly these four lines and nothing else:
CORRECT: <correct | partly correct | wrong>
SUPPORTED: <yes | no>
REASON: <one sentence explaining the two verdicts>
UNSUPPORTED: <exact quote of the first unsupported statement, or none>"""


def format_chunks(chunk_list):
    """chunk_list: list of Chunk objects -> numbered block [1] ... [n]."""
    return "\n\n".join(f"[{i}] (source: {c.source})\n{' '.join(c.text.split())}"
                       for i, c in enumerate(chunk_list, start=1))


def gen_prompt(question, chunk_list):
    return GEN_TEMPLATE.format(chunks=format_chunks(chunk_list), question=question)


def judge_prompt(question, expected, chunk_list, answer):
    return JUDGE_TEMPLATE.format(question=question, expected=expected,
                                 chunks=format_chunks(chunk_list), answer=answer)


# ----------------------------------------------------------------------------- models
def free():
    """Call after `del model` to release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_instruct(repo):
    """Pre-quantised 4-bit (bitsandbytes) instruct model from the Hub, loaded straight to the GPU."""
    tok = AutoTokenizer.from_pretrained(repo)
    kwargs = ({"device_map": {"": 0}} if torch.cuda.is_available() else {"dtype": torch.float32})
    model = AutoModelForCausalLM.from_pretrained(repo, low_cpu_mem_usage=True, **kwargs)
    model.eval()
    return model, tok


def load_a1_model(base=config.A1_BASE, adapter=config.A1_ADAPTER):
    """Assignment 1 model: CPT checkpoint (4-bit NF4, as in QLoRA training) + Adapter B."""
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(adapter)  # saved with the TinyLlama-Chat (Zephyr) template
    if torch.cuda.is_available():
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tok


@torch.no_grad()
def chat(model, tok, system, user, max_new_tokens=config.GEN_MAX_NEW_TOKENS):
    """Greedy decoding (= temperature 0): the same input always gives the same output."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------- answer checks
def cites(answer):
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def says_not_found(answer):
    return config.NOT_FOUND.lower() in answer.lower()


def parse_judge(text):
    def grab(label):
        m = re.search(rf"^\s*\**{label}\**\s*:\s*(.+)$", text, flags=re.I | re.M)
        return m.group(1).strip().strip("*").strip() if m else ""
    correct = grab("CORRECT").lower()
    correct = ("partly correct" if "partly" in correct or "partial" in correct
               else "wrong" if "wrong" in correct or "incorrect" in correct
               else "correct" if "correct" in correct else "unparsed")
    supported = grab("SUPPORTED").lower()
    supported = "yes" if supported.startswith("yes") else "no" if supported.startswith("no") else "unparsed"
    return {"correct": correct, "supported": supported, "reason": grab("REASON"),
            "unsupported quote": grab("UNSUPPORTED")}


# ----------------------------------------------------------------------------- disk (lab volumes are small)
def hf_cache_dir():
    return Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")


def free_gb(path=None):
    path = Path(path or hf_cache_dir())
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 2**30


def delete_cached(repo):
    """Removes one model from the Hugging Face cache (used to make room for the judge)."""
    target = hf_cache_dir() / ("models--" + repo.replace("/", "--"))
    if target.exists():
        shutil.rmtree(target)
        return True
    return False
