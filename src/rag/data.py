"""Corpus loading, fixed-size token chunking, queries and the hit test."""
import json
import re
from dataclasses import dataclass

from src.rag import config


def norm(text: str) -> str:
    """Lower-case and collapse whitespace (PDF line breaks) so spans match across lines."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_corpus(corpus_dir=config.CORPUS_DIR):
    """{file name: text} for every cleaned .txt document from Assignment 1."""
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.txt"))}
    if not docs:
        raise FileNotFoundError(f"No .txt files in {corpus_dir}")
    return docs


def load_queries(path=config.QUERIES):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"], data["out_of_corpus"]


@dataclass
class Chunk:
    chunk_id: int
    source: str
    text: str
    n_tokens: int


def chunk_fixed(docs: dict, tokenizer, size: int):
    """Fixed-size chunking: every `size` tokens of the embedding model's tokenizer, no overlap.

    Chunks never cross document boundaries (the last chunk of a document may be shorter).
    Chunk text is cut from the original document via the tokenizer's character offsets,
    so it is exactly the corpus text (needed for the answer-span hit test).
    """
    chunks = []
    for source, text in docs.items():
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        offsets = enc["offset_mapping"]
        for start in range(0, len(offsets), size):
            piece = offsets[start:start + size]
            chunk_text = text[piece[0][0]:piece[-1][1]]
            chunks.append(Chunk(len(chunks), source, chunk_text, len(piece)))
    return chunks


def is_hit(chunk_text: str, span: str) -> bool:
    """A retrieved chunk is a hit if it contains the query's whole answer span."""
    return norm(span) in norm(chunk_text)


def answer_chunk_ids(chunks, span):
    """Chunks that contain the span (empty if the span straddles a chunk boundary)."""
    return [c.chunk_id for c in chunks if is_hit(c.text, span)]
