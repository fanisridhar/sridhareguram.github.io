"""Semantic retrieval service with switchable optimisations.

Pipeline: query -> embed (TF-IDF + truncated SVD "LSA" embedding)
          -> score vs. all docs -> top-k selection -> fetch doc records
          [-> optional lexical BM25 leg + reciprocal-rank fusion]

The *baseline* configuration deliberately uses the straightforward code most
people write first (sklearn cosine_similarity on every call, full argsort,
linear scan to fetch records). Every optimisation is a flag, so each change can
be measured on its own against the same fixed benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tracer import Trace


@dataclass(frozen=True)
class Config:
    name: str
    precomputed_norm_f32: bool = False   # cache normalised float32 doc matrix
    partial_topk: bool = False           # argpartition instead of full argsort
    dict_lookup: bool = False            # O(1) record fetch instead of list scan
    hybrid: bool = False                 # add BM25 leg + reciprocal rank fusion
    k: int = 5
    fetch_n: int = 50                    # records fetched for reranking/display


class RetrievalService:
    def __init__(self, docs, dim=128, seed=0):
        self.docs = docs
        self.records = [{"id": i, "text": d} for i, d in enumerate(docs)]       # list scan target
        self.by_id = {r["id"]: r for r in self.records}                         # dict lookup target
        self.tfidf = TfidfVectorizer(sublinear_tf=True)
        X = self.tfidf.fit_transform(docs)
        self.svd = TruncatedSVD(n_components=dim, random_state=seed)
        self.E64 = self.svd.fit_transform(X)                                     # float64, un-normalised
        n = np.linalg.norm(self.E64, axis=1, keepdims=True)
        self.E32 = (self.E64 / np.maximum(n, 1e-9)).astype(np.float32)           # normalised float32
        # BM25 index (sparse)
        self._build_bm25()

    # ---- BM25 ------------------------------------------------------------
    def _build_bm25(self, k1=1.5, b=0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        self.cv = CountVectorizer()
        tf = self.cv.fit_transform(self.docs).tocsr().astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avg = dl.mean()
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        N = tf.shape[0]
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
        w = (tf.data * (k1 + 1) / denom) * self.idf[tf.col]
        self.bm25 = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).tocsc()

    def _bm25_scores(self, text):
        q = self.cv.transform([text])
        cols = q.indices
        if len(cols) == 0:
            return np.zeros(len(self.docs), dtype=np.float32)
        return np.asarray(self.bm25[:, cols].sum(axis=1)).ravel()

    # ---- query path ------------------------------------------------------
    def embed(self, text):
        return self.svd.transform(self.tfidf.transform([text]))   # (1, dim) float64

    def search(self, text, cfg: Config, trace: Trace | None = None):
        tr = trace or Trace(-1, text)
        with tr.span("embed"):
            q64 = self.embed(text)
        with tr.span("score"):
            if cfg.precomputed_norm_f32:
                q = (q64 / max(np.linalg.norm(q64), 1e-9)).astype(np.float32)
                scores = (self.E32 @ q.T).ravel()
            else:
                scores = cosine_similarity(q64, self.E64).ravel()   # re-normalises all docs each call
        with tr.span("topk"):
            n = max(cfg.fetch_n, cfg.k)
            if cfg.partial_topk:
                idx = np.argpartition(-scores, n)[:n]
                idx = idx[np.argsort(-scores[idx])]
            else:
                idx = np.argsort(-scores)[:n]
        if cfg.hybrid:
            with tr.span("lexical"):
                lex = self._bm25_scores(text)
                lidx = np.argpartition(-lex, 100)[:100]
                lidx = lidx[np.argsort(-lex[lidx])]
            with tr.span("fuse"):
                dense_top = np.argpartition(-scores, 100)[:100]
                dense_top = dense_top[np.argsort(-scores[dense_top])]
                fused: dict = {}
                for rank, d in enumerate(dense_top):
                    fused[int(d)] = fused.get(int(d), 0.0) + 1.0 / (60 + rank)
                for rank, d in enumerate(lidx):
                    fused[int(d)] = fused.get(int(d), 0.0) + 1.0 / (60 + rank)
                idx = np.array(sorted(fused, key=fused.get, reverse=True)[:n])
        with tr.span("fetch"):
            if cfg.dict_lookup:
                recs = [self.by_id[int(i)] for i in idx[: cfg.fetch_n]]
            else:
                recs = [next(r for r in self.records if r["id"] == int(i)) for i in idx[: cfg.fetch_n]]
        top = [r["id"] for r in recs[: cfg.k]]
        tr.finish()
        return top, tr
