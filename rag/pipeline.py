"""End-to-end flow:  message -> safety check -> retrieval (ChromaDB + BM25 + TF-IDF) -> local LLM -> reply.

Knowledge base = the 350 MentalQA training cases, minus cases whose question mentions suicide or
self-harm. Those are excluded so a vulnerable user is never shown them as a "similar case";
crisis messages are handled by the safety layer instead.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from . import config
from .dataset import QUESTION_TYPES, Case, knowledge_base
from .embeddings import LOAD_ERRORS, load_embedder
from .generation import Reply, generate
from .retrieval import Hit, Retriever
from .safety import is_crisis
from .units import make_units


def safe_cases() -> list[Case]:
    return [c for c in knowledge_base() if not is_crisis(c.question)]


def production_embedder():
    emb = load_embedder(config.EMBEDDING_MODEL)
    if emb is None:
        return load_embedder("lsa"), LOAD_ERRORS.get(config.EMBEDDING_MODEL, "")
    return emb, ""


@lru_cache(maxsize=1)
def assistant_index() -> tuple[Retriever, dict]:
    t0 = time.perf_counter()
    cases = safe_cases()
    emb, fallback = production_embedder()
    retriever = Retriever(make_units(cases, config.UNIT), emb, persistent=True)
    info = {"cases": len(cases), "excluded_crisis_cases": len(knowledge_base()) - len(cases), "unit": config.UNIT,
            "embedder": emb.name, "fallback_reason": fallback, "vectors": retriever.index.count(),
            "build_seconds": round(time.perf_counter() - t0, 1)}
    return retriever, info


def predict_types(hits: list[Hit], min_share: float = 0.5) -> list[str]:
    """Question type by voting over the retrieved cases (a type wins if at least half of them have it)."""
    votes = Counter(t for h in hits for t in h.chunk.case.types)
    chosen = [t for t, v in votes.most_common() if v / max(1, len(hits)) >= min_share]
    return chosen or ([votes.most_common(1)[0][0]] if votes else [])


@dataclass
class Result:
    message: str
    crisis: bool
    reply: Reply | None = None
    types: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0

    @property
    def type_names(self) -> list[str]:
        return [QUESTION_TYPES.get(t, t) for t in self.types]


def respond(message: str, use_llm: bool = True, k: int = config.TOP_K) -> Result:
    if is_crisis(message):
        return Result(message, crisis=True)
    retriever, _ = assistant_index()
    t0 = time.perf_counter()
    hits = retriever.search(message, k=k)
    r_ms = (time.perf_counter() - t0) * 1000
    reply = generate(message, hits, use_llm=use_llm)
    return Result(message, False, reply, predict_types(hits), r_ms)
