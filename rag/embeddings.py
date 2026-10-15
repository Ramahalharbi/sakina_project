"""Embedding models (all free, no API key).

    intfloat/multilingual-e5-base    default. Downloaded once from Hugging Face (~1.1 GB), then runs
                                     offline on your computer. Uses e5's "query: " / "passage: " prefixes.
    intfloat/multilingual-e5-small   smaller and faster variant (~470 MB), useful for comparison.
    lsa                              offline fallback: character n-gram TF-IDF reduced with SVD.
                                     Used automatically when a model cannot be downloaded, and always
                                     available as a baseline in the evaluation.
Vectors are L2-normalized, so the dot product equals cosine similarity.
"""
from __future__ import annotations

import numpy as np

from .arabic import normalize


def _l2(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


class Embedder:
    name = ""
    needs_fit = False

    def fit(self, texts: list[str]):
        return self

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError


class E5Embedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, is_query=False):
        prefix = "query: " if is_query else "passage: "
        return _l2(self.model.encode([prefix + t for t in texts], batch_size=32, show_progress_bar=False,
                                     normalize_embeddings=True))


class LSAEmbedder(Embedder):
    name = "lsa"
    needs_fit = True

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vectorizer = self.svd = None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        x = self.vectorizer.fit_transform([normalize(t) for t in texts])
        k = max(2, min(self.dim, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(x)
        return self

    def encode(self, texts, is_query=False):
        return _l2(self.svd.transform(self.vectorizer.transform([normalize(t) for t in texts])))


_CACHE: dict[str, Embedder] = {}
LOAD_ERRORS: dict[str, str] = {}


def load_embedder(name: str) -> Embedder | None:
    """Return the embedder, or None (with the reason in LOAD_ERRORS) if it cannot be loaded."""
    if name == "lsa":
        return LSAEmbedder()
    if name in _CACHE:
        return _CACHE[name]
    if name in LOAD_ERRORS:          # do not retry a failed download on every call
        return None
    try:
        _CACHE[name] = E5Embedder(name)
        return _CACHE[name]
    except Exception as exc:  # no internet on first run, package missing, ...
        LOAD_ERRORS[name] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return None
