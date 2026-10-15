"""Vector database (ChromaDB) and hybrid retrieval over knowledge-base units.

    dense    cosine similarity search in ChromaDB over the embedded chunks
    bm25     BM25 over normalized, lightly stemmed Arabic tokens (exact words)
    tfidf    character n-gram TF-IDF (robust to Arabic prefixes and spelling variants)
    hybrid   Reciprocal Rank Fusion of the three: score = sum of 1 / (60 + rank)
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import config
from .arabic import normalize, tokenize
from .units import Unit as Chunk
from .embeddings import Embedder

METHODS = ["dense", "bm25", "tfidf", "hybrid"]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class ChromaIndex:
    """One ChromaDB collection per (chunking, embedding) configuration."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, persistent: bool):
        import chromadb
        from chromadb.config import Settings

        settings = Settings(anonymized_telemetry=False)
        self.client = (chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"), settings=settings)
                       if persistent else chromadb.EphemeralClient(settings=settings))
        sig = hashlib.md5(("|".join(c.chunk_id for c in chunks) + embedder.name).encode()).hexdigest()[:10]
        name = f"kb_{sig}" if persistent else f"tmp_{uuid.uuid4().hex[:10]}"
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self.embedder = embedder
        self.embed_seconds = 0.0
        have = set(self.col.get(include=[])["ids"])
        todo = [c for c in chunks if c.chunk_id not in have]
        if todo:
            t0 = time.perf_counter()
            vecs = embedder.encode([c.embed_text for c in todo])
            self.embed_seconds = time.perf_counter() - t0
            for i in range(0, len(todo), 1000):
                part = todo[i:i + 1000]
                self.col.add(ids=[c.chunk_id for c in part], embeddings=vecs[i:i + 1000].tolist(),
                             documents=[c.text for c in part],
                             metadatas=[{"case_id": c.case.case_id, "unit": c.mode} for c in part])
        self.reused = not todo

    def count(self) -> int:
        return self.col.count()

    def search(self, query: str, n: int) -> dict[str, float]:
        q = self.embedder.encode([query], is_query=True)[0]
        res = self.col.query(query_embeddings=[q.tolist()], n_results=min(n, self.count()), include=["distances"])
        return {cid: 1.0 - d for cid, d in zip(res["ids"][0], res["distances"][0])}


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None, persistent: bool = False):
        self.chunks = chunks
        self.pos = {c.chunk_id: i for i, c in enumerate(chunks)}
        texts = [c.embed_text for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in texts])
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.tfidf_m = self.tfidf.fit_transform([normalize(t) for t in texts])
        self.index = None
        if embedder is not None:
            if embedder.needs_fit:
                embedder.fit(texts)
            self.index = ChromaIndex(chunks, embedder, persistent)

    @property
    def methods(self) -> list[str]:
        return [m for m in METHODS if m in ("bm25", "tfidf") or self.index is not None]

    def _scores(self, query: str, method: str) -> np.ndarray:
        if method == "bm25":
            return np.asarray(self.bm25.get_scores(tokenize(query) or ["_"]), dtype=np.float64)
        if method == "tfidf":
            return linear_kernel(self.tfidf.transform([normalize(query)]), self.tfidf_m).ravel()
        if method == "dense":
            s = np.full(len(self.chunks), -1.0)
            for cid, score in self.index.search(query, len(self.chunks)).items():
                s[self.pos[cid]] = score
            return s
        raise ValueError(method)

    def search(self, query: str, k: int = config.TOP_K, method: str = config.RETRIEVAL_METHOD) -> list[Hit]:
        if method == "hybrid":
            fused = np.zeros(len(self.chunks))
            for m in [x for x in self.methods if x != "hybrid"]:
                ranks = np.empty(len(self.chunks), dtype=np.int64)
                ranks[np.argsort(-self._scores(query, m))] = np.arange(1, len(self.chunks) + 1)
                fused += 1.0 / (config.RRF_K + ranks)
            scores = fused
        else:
            scores = self._scores(query, method)
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), r + 1) for r, i in enumerate(order)]
