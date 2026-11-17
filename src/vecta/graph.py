"""HNSW graph index (Malkov & Yashunin, 2018) and RaBitQ-style binary quantization with re-ranking."""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from vecta.kernels import KernelBridge, get_bridge
from vecta.quantize import prepare

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
U8 = npt.NDArray[np.uint8]
U64 = npt.NDArray[np.uint64]
Filter = Callable[[int], bool]


class HNSWIndex:
    """Hierarchical Navigable Small World graph.

    * exponentially-decaying random levels with ``mL = 1 / ln(M)``;
    * greedy descent through upper layers, beam search (``ef``) on the target layers;
    * the neighbour-selection **heuristic** (paper Algorithm 4), which keeps a candidate only
      if it is closer to the query than to every already-selected neighbour, preserving
      long-range links between clusters;
    * ``M`` links per node on upper layers and ``2M`` on layer 0;
    * tombstone deletion and predicate-filtered search, where filtered-out or deleted nodes
      are still traversed but never returned, so connectivity is preserved.
    """

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 100,
        metric: str = "l2",
        seed: int = 0,
        bridge: KernelBridge | None = None,
    ) -> None:
        if m < 2:
            raise ValueError("m must be >= 2")
        prepare(np.zeros((1, dim)), metric)
        self.dim, self.m, self.ef_construction, self.metric = dim, m, ef_construction, metric
        self.seed = seed
        self.m_max0 = 2 * m
        self.ml = 1.0 / math.log(m)
        self.rng = np.random.default_rng(seed)
        self.bridge = bridge or get_bridge()
        self._vec: F32 = np.zeros((16, dim), dtype=np.float32)
        self.count = 0
        self.levels: list[int] = []
        self.links: list[list[list[int]]] = []
        self.deleted: set[int] = set()
        self.entry = -1
        self.max_level = -1
        self.distance_evals = 0

    # --------------------------------------------------------------------------- utilities

    def __len__(self) -> int:
        return self.count - len(self.deleted)

    @property
    def vectors(self) -> F32:
        return self._vec[: self.count]

    def _dists(self, query: F32, ids: list[int]) -> F64:
        self.distance_evals += len(ids)
        return self.bridge.l2_gather(query, self._vec, np.asarray(ids, dtype=np.int64))

    def _random_level(self) -> int:
        return int(-math.log(1.0 - float(self.rng.random())) * self.ml)

    def _ensure_capacity(self, extra: int) -> None:
        need = self.count + extra
        if need > self._vec.shape[0]:
            cap = max(need, 2 * self._vec.shape[0])
            grown = np.zeros((cap, self.dim), dtype=np.float32)
            grown[: self.count] = self._vec[: self.count]
            self._vec = grown

    # ---------------------------------------------------------------------------- search core

    def _search_layer(
        self,
        query: F32,
        entry: list[tuple[float, int]],
        ef: int,
        layer: int,
        admit: Filter | None = None,
    ) -> list[tuple[float, int]]:
        """Beam search on one layer. Returns up to ``ef`` admitted (dist, id) pairs, ascending."""
        visited = {i for _, i in entry}
        candidates = list(entry)
        heapq.heapify(candidates)
        results: list[tuple[float, int]] = []  # max-heap via negation
        for d, i in entry:
            if admit is None or admit(i):
                heapq.heappush(results, (-d, -i))
        while candidates:
            d, i = heapq.heappop(candidates)
            if len(results) >= ef and d > -results[0][0]:
                break
            fresh = [n for n in self.links[i][layer] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for nd, n in zip(self._dists(query, fresh).tolist(), fresh, strict=True):
                if len(results) < ef or nd < -results[0][0]:
                    heapq.heappush(candidates, (nd, n))
                    if admit is None or admit(n):
                        heapq.heappush(results, (-nd, -n))
                        if len(results) > ef:
                            heapq.heappop(results)
        return sorted((-d, -i) for d, i in results)

    def _select_heuristic(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        selected: list[int] = []
        for d, c in sorted(candidates):
            if len(selected) >= m:
                break
            if not selected or bool(np.all(self._dists(self._vec[c], selected) > d)):
                selected.append(c)
        return selected

    # ------------------------------------------------------------------------------- mutation

    def add(self, x: npt.ArrayLike) -> I64:
        arr = prepare(x, self.metric)
        if arr.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {arr.shape[1]}")
        self._ensure_capacity(arr.shape[0])
        ids = np.arange(self.count, self.count + arr.shape[0], dtype=np.int64)
        for row in arr:
            self._insert(row)
        return ids

    def _insert(self, vec: F32) -> None:
        node = self.count
        self._vec[node] = vec
        self.count += 1
        level = self._random_level()
        self.levels.append(level)
        self.links.append([[] for _ in range(level + 1)])
        query = self._vec[node]
        if self.entry < 0:
            self.entry, self.max_level = node, level
            return
        ep = [(float(self._dists(query, [self.entry])[0]), self.entry)]
        for layer in range(self.max_level, level, -1):
            ep = self._search_layer(query, ep, 1, layer)[:1]
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(query, ep, self.ef_construction, layer)
            cap = self.m_max0 if layer == 0 else self.m
            neighbours = self._select_heuristic(found, self.m)
            self.links[node][layer] = neighbours
            for nb in neighbours:
                nb_links = self.links[nb][layer]
                nb_links.append(node)
                if len(nb_links) > cap:
                    dists = self._dists(self._vec[nb], nb_links).tolist()
                    self.links[nb][layer] = self._select_heuristic(
                        list(zip(dists, nb_links, strict=True)), cap
                    )
            ep = found
        if level > self.max_level:
            self.entry, self.max_level = node, level

    def delete(self, ids: npt.ArrayLike) -> None:
        for i in np.atleast_1d(np.asarray(ids, dtype=np.int64)).tolist():
            if not 0 <= i < self.count:
                raise IndexError(f"id {i} out of range")
            self.deleted.add(int(i))

    # --------------------------------------------------------------------------------- search

    def search(
        self,
        queries: npt.ArrayLike,
        k: int = 10,
        ef: int = 64,
        filter_fn: Filter | None = None,
    ) -> tuple[I64, F64]:
        q = prepare(queries, self.metric)
        out_ids = np.full((q.shape[0], k), -1, dtype=np.int64)
        out_d = np.full((q.shape[0], k), np.inf, dtype=np.float64)
        if self.entry < 0:
            return out_ids, out_d
        deleted = self.deleted

        def admit(i: int) -> bool:
            return i not in deleted and (filter_fn is None or filter_fn(i))

        use_admit = admit if (deleted or filter_fn is not None) else None
        for qi in range(q.shape[0]):
            query = q[qi]
            ep = [(float(self._dists(query, [self.entry])[0]), self.entry)]
            for layer in range(self.max_level, 0, -1):
                ep = self._search_layer(query, ep, 1, layer)[:1]
            found = self._search_layer(query, ep, max(ef, k), 0, use_admit)[:k]
            for j, (d, i) in enumerate(found):
                out_ids[qi, j] = i
                out_d[qi, j] = d
        return out_ids, out_d

    # ---------------------------------------------------------------------------- statistics

    def graph_stats(self) -> dict[str, float]:
        degrees = [len(lk[0]) for lk in self.links]
        reach = self._reachable_from_entry()
        return {
            "nodes": float(self.count),
            "max_level": float(self.max_level),
            "mean_degree_l0": float(np.mean(degrees)) if degrees else 0.0,
            "max_degree_l0": float(max(degrees, default=0)),
            "reachable_fraction": reach / self.count if self.count else 1.0,
        }

    def _reachable_from_entry(self) -> int:
        if self.entry < 0:
            return 0
        seen = {self.entry}
        stack = [self.entry]
        while stack:
            for n in self.links[stack.pop()][0]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen)

    # ---------------------------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        flat: list[int] = []
        offsets = [0]
        for node_links in self.links:
            for layer_links in node_links:
                flat.extend(layer_links)
                offsets.append(len(flat))
        meta = {
            "dim": self.dim,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "metric": self.metric,
            "seed": self.seed,
            "entry": self.entry,
            "max_level": self.max_level,
        }
        np.savez_compressed(
            path,
            vectors=self.vectors,
            levels=np.asarray(self.levels, dtype=np.int64),
            flat=np.asarray(flat, dtype=np.int64),
            offsets=np.asarray(offsets, dtype=np.int64),
            deleted=np.asarray(sorted(self.deleted), dtype=np.int64),
            meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: Path, bridge: KernelBridge | None = None) -> HNSWIndex:
        with np.load(path) as data:
            meta = json.loads(bytes(data["meta"]).decode())
            idx = cls(meta["dim"], meta["m"], meta["ef_construction"], meta["metric"], meta["seed"], bridge)
            vectors = data["vectors"]
            idx._vec = np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1, meta["dim"])
            if idx._vec.shape[0] == 0:
                idx._vec = np.zeros((16, meta["dim"]), dtype=np.float32)
            idx.count = int(vectors.shape[0])
            idx.levels = data["levels"].tolist()
            flat, offsets = data["flat"].tolist(), data["offsets"].tolist()
            idx.deleted = set(data["deleted"].tolist())
        pos = 0
        idx.links = []
        for level in idx.levels:
            node_links = []
            for _ in range(level + 1):
                node_links.append(flat[offsets[pos] : offsets[pos + 1]])
                pos += 1
            idx.links.append(node_links)
        idx.entry, idx.max_level = meta["entry"], meta["max_level"]
        return idx


class BinaryIndex:
    """RaBitQ-flavoured 1-bit quantization: random orthogonal rotation, sign bits, Hamming
    shortlist via hardware popcount, then exact re-ranking of the shortlist.

    Codes cost ``dim / 8`` bytes per vector (32x smaller than float32). The rotation spreads
    variance across coordinates so that each sign bit carries comparable information.
    """

    def __init__(
        self, dim: int, metric: str = "l2", seed: int = 0, bridge: KernelBridge | None = None
    ) -> None:
        prepare(np.zeros((1, dim)), metric)
        self.dim, self.metric = dim, metric
        self.words = (dim + 63) // 64
        gauss = np.random.default_rng(seed).normal(size=(dim, dim))
        q, r = np.linalg.qr(gauss)
        self.rotation: F32 = (q * np.sign(np.diag(r))).astype(np.float32)
        self.mean: F32 = np.zeros(dim, dtype=np.float32)
        self.bridge = bridge or get_bridge()
        self.codes: U64 = np.zeros((0, self.words), dtype=np.uint64)
        self.raw: F32 = np.zeros((0, dim), dtype=np.float32)
        self.trained = False

    def train(self, x: npt.ArrayLike) -> BinaryIndex:
        self.mean = prepare(x, self.metric).mean(axis=0).astype(np.float32)
        self.trained = True
        return self

    def encode(self, x: F32) -> U64:
        proj = (x - self.mean) @ self.rotation.T
        bits = np.packbits(proj > 0, axis=1, bitorder="little")
        padded = np.zeros((x.shape[0], self.words * 8), dtype=np.uint8)
        padded[:, : bits.shape[1]] = bits
        out: U64 = padded.view(np.uint64).reshape(x.shape[0], self.words)
        return out

    def add(self, x: npt.ArrayLike) -> I64:
        if not self.trained:
            raise RuntimeError("train the index first")
        arr = prepare(x, self.metric)
        start = self.raw.shape[0]
        self.codes = np.ascontiguousarray(np.vstack([self.codes, self.encode(arr)]))
        self.raw = np.ascontiguousarray(np.vstack([self.raw, arr]))
        return np.arange(start, start + arr.shape[0], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.raw.shape[0])

    @property
    def code_bytes_per_vector(self) -> int:
        return self.words * 8

    def search(self, queries: npt.ArrayLike, k: int = 10, rerank: int = 100) -> tuple[I64, F64]:
        q = prepare(queries, self.metric)
        shortlist = max(k, rerank)
        out_ids = np.full((q.shape[0], k), -1, dtype=np.int64)
        out_d = np.full((q.shape[0], k), np.inf, dtype=np.float64)
        qcodes = self.encode(q)
        for qi in range(q.shape[0]):
            ids, _ = self.bridge.hamming_topk(self.codes, qcodes[qi], shortlist)
            ids = ids[ids >= 0]
            exact = self.bridge.l2_gather(q[qi], self.raw, ids)
            order = np.lexsort((ids, exact))[:k]
            out_ids[qi, : order.size] = ids[order]
            out_d[qi, : order.size] = exact[order]
        return out_ids, out_d
