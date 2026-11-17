"""Hybrid lexical + vector retrieval: Okapi BM25 and Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 (Robertson & Zaragoza, 2009) with the Lucene non-negative IDF."""

    def __init__(self, docs: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        if not docs:
            raise ValueError("BM25 needs at least one document")
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.avgdl = sum(self.lengths) / len(docs) or 1.0
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out: list[float] = []
        for tf, dl in zip(self.tf, self.lengths, strict=True):
            s = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1.0) / (f + norm)
            out.append(s)
        return out

    def search(self, query: str, k: int = 10) -> list[int]:
        sc = self.scores(query)
        ranked = sorted((i for i in range(len(sc)) if sc[i] > 0), key=lambda i: (-sc[i], i))
        return ranked[:k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 10, c: float = 60.0
) -> list[tuple[int, float]]:
    """RRF (Cormack et al., 2009): score(d) = sum_r 1 / (c + rank_r(d)), ranks starting at 1."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            if doc < 0:
                continue
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (c + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
