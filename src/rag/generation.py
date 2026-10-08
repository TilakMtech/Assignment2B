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
2. The chunks are labelled [1], [2] and [3]. After each sentence, cite the label(s) of the chunk(s) you used, for example [2]. Cite ONLY these labels: numbers that appear inside the chunk text, such as 5.2 or 4.1.1, are policy section numbers, not chunk labels, and must never be cited.
3. If the chunks do not contain the answer, reply exactly: Not found in the documents.
4. Answer in 1 to 3 sentences.

Chunks:
{chunks}

Question: {question}
Answer:"""

# Version used for the saved Assignment 1 trial (before the citation-label sentence was added).
GEN_TEMPLATE_V1 = """Answer the question using ONLY the numbered chunks below.

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


# ----------------------------------------------------------------------------- judge decoding
JUDGE_LABELS = {"CORRECT": ["correct", "partly correct", "wrong"], "SUPPORTED": ["yes", "no"]}


def _first_token_logprob(model, tok, context, option):
    """Log-probability of the first token of ' option' after `context`."""
    ctx = tok(context, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    first = tok(" " + option, add_special_tokens=False)["input_ids"][0]
    logits = model(input_ids=ctx).logits[0, -1].float()
    return torch.log_softmax(logits, -1)[first].item()


def _continue_line(model, tok, context, max_new_tokens=80):
    ids = tok(context, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()


@torch.no_grad()
def judge(model, tok, user_prompt, system=None):
    """Runs the judge prompt with constrained, greedy decoding of the fixed output format.

    Small instruct models often leave the label fields empty (seen with Llama-3.2-3B), so the
    assistant reply is built line by line: for CORRECT and SUPPORTED the judge picks the allowed
    label with the highest probability (argmax = temperature 0); REASON and UNSUPPORTED are then
    generated greedily, conditioned on the chosen labels. Returns the parsed verdict and raw text.
    """
    system = system or JUDGE_SYSTEM
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    context = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    reply = ""
    verdict = {}
    for field, options in JUDGE_LABELS.items():
        prefix = context + reply + f"{field}:"
        scores = {o: _first_token_logprob(model, tok, prefix, o) for o in options}
        verdict[field.lower()] = max(scores, key=scores.get)
        reply += f"{field}: {verdict[field.lower()]}\n"
    reply += "REASON:"
    verdict["reason"] = _continue_line(model, tok, context + reply)
    reply += f" {verdict['reason']}\nUNSUPPORTED:"
    verdict["unsupported quote"] = _continue_line(model, tok, context + reply)
    reply += f" {verdict['unsupported quote']}"
    return verdict, reply


# ----------------------------------------------------------------------------- answer checks
def cites(answer):
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def citation_check(answer, n_chunks=3):
    """Splits bracketed citations into valid chunk labels [1..n] and anything else (e.g. [5.2])."""
    found = re.findall(r"\[([^\[\]]{1,12})\]", answer)
    valid = sorted({int(c) for c in found if c.strip().isdigit() and 1 <= int(c) <= n_chunks})
    invalid = sorted({c.strip() for c in found if not (c.strip().isdigit() and 1 <= int(c) <= n_chunks)})
    return valid, invalid


def a1_tokenizer():
    """Tokenizer of the Assignment 1 model with the chat template it was fine-tuned with."""
    from transformers import AutoTokenizer
    src = config.A1_ADAPTER if (config.A1_ADAPTER / "tokenizer_config.json").exists() else config.A1_TOKENIZER_HUB
    tok = AutoTokenizer.from_pretrained(src)
    tok.chat_template = config.A1_CHAT_TEMPLATE
    return tok


def chat_prompt_tokens(tok, system, user):
    """Exact number of tokens of the fully formatted chat prompt (what the model actually receives)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return len(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])


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


def cached(repo):
    return (hf_cache_dir() / ("models--" + repo.replace("/", "--"))).exists()


def repo_size_gb(repo, default=6.0):
    """Download size of a Hub model (weights + tokenizer); `default` if the Hub cannot be asked."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        return sum((f.size or 0) for f in info.siblings) / 2**30
    except Exception:
        return default


def make_room(repo, drop=None, margin_gb=0.5):
    """Checks there is space to download `repo`; if not and `drop` is given, deletes `drop` from the
    cache first. Raises SystemExit with a clear message if it still does not fit."""
    if cached(repo):
        return
    need = repo_size_gb(repo) + margin_gb
    if free_gb() < need and drop and delete_cached(drop):
        print(f"Removed {drop} from the cache to make room.")
    if free_gb() < need:
        raise SystemExit(f"{repo} needs ~{need:.1f} GiB but only {free_gb():.1f} GiB is free in {hf_cache_dir()}. "
                         "Free space or point HF_HOME to a bigger volume, then re-run.")


def delete_cached(repo):
    """Removes one model from the Hugging Face cache (used to make room for the judge)."""
    target = hf_cache_dir() / ("models--" + repo.replace("/", "--"))
    if target.exists():
        shutil.rmtree(target)
        return True
    return False
