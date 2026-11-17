"""Exact flat index, k-means++ / Lloyd clustering, product quantization and IVF-PQ."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from vecta.kernels import KernelBridge, get_bridge

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
U8 = npt.NDArray[np.uint8]

METRICS = ("l2", "cosine")


def prepare(x: npt.ArrayLike, metric: str) -> F32:
    """Cast to contiguous float32 and L2-normalise for cosine (cosine ranking == L2 on unit vectors)."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    arr = np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=np.float32)))
    if metric == "cosine":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = np.ascontiguousarray(arr / np.where(norms == 0, 1.0, norms), dtype=np.float32)
    return arr


class FlatIndex:
    """Exact brute-force search (the ground-truth oracle) with tombstone deletes."""

    def __init__(self, dim: int, metric: str = "l2", bridge: KernelBridge | None = None) -> None:
        prepare(np.zeros((1, dim)), metric)
        self.dim = dim
        self.metric = metric
        self.bridge = bridge or get_bridge()
        self.vectors: F32 = np.zeros((0, dim), dtype=np.float32)
        self.alive: U8 = np.zeros(0, dtype=np.uint8)

    def __len__(self) -> int:
        return int(self.alive.sum())

    def add(self, x: npt.ArrayLike) -> I64:
        arr = prepare(x, self.metric)
        if arr.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {arr.shape[1]}")
        start = self.vectors.shape[0]
        self.vectors = np.ascontiguousarray(np.vstack([self.vectors, arr]))
        self.alive = np.concatenate([self.alive, np.ones(arr.shape[0], dtype=np.uint8)])
        return np.arange(start, start + arr.shape[0], dtype=np.int64)

    def delete(self, ids: npt.ArrayLike) -> None:
        self.alive[np.asarray(ids, dtype=np.int64)] = 0

    def search(self, queries: npt.ArrayLike, k: int = 10) -> tuple[I64, F64]:
        q = prepare(queries, self.metric)
        mask = None if self.alive.all() else self.alive
        return self.bridge.l2_topk(q, self.vectors, k, mask)


def sqdist(x: npt.NDArray[np.floating], c: npt.NDArray[np.floating]) -> F64:
    """Pairwise squared L2 via ||x||^2 - 2 x.c + ||c||^2 (float64, clipped at 0)."""
    x64 = x.astype(np.float64)
    c64 = c.astype(np.float64)
    d = (x64 * x64).sum(axis=1)[:, None] - 2.0 * (x64 @ c64.T) + (c64 * c64).sum(axis=1)[None, :]
    out: F64 = np.maximum(d, 0.0)
    return out


def kmeans(x: F32, k: int, iters: int = 20, seed: int = 0) -> F32:
    """k-means++ seeding (Arthur & Vassilvitskii, 2007) followed by Lloyd iterations."""
    n = x.shape[0]
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= number of points")
    rng = np.random.default_rng(seed)
    x64 = x.astype(np.float64)
    centres = np.empty((k, x.shape[1]), dtype=np.float64)
    centres[0] = x64[rng.integers(n)]
    closest = ((x64 - centres[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        total = closest.sum()
        idx = int(rng.choice(n, p=closest / total)) if total > 0 else int(rng.integers(n))
        centres[c] = x64[idx]
        closest = np.minimum(closest, ((x64 - centres[c]) ** 2).sum(axis=1))
    for _ in range(iters):
        d = sqdist(x64, centres)
        assign = d.argmin(axis=1)
        moved = False
        for c in range(k):
            members = x64[assign == c]
            if members.shape[0] == 0:
                # Re-seed an empty cluster at the point farthest from its centre.
                far = int(d[np.arange(n), assign].argmax())
                new = x64[far]
            else:
                new = members.mean(axis=0)
            moved |= not np.allclose(new, centres[c])
            centres[c] = new
        if not moved:
            break
    out: F32 = centres.astype(np.float32)
    return out


def assign_nearest(x: F32, centres: F32, chunk: int = 4096) -> I64:
    out = np.empty(x.shape[0], dtype=np.int64)
    for start in range(0, x.shape[0], chunk):
        out[start : start + chunk] = sqdist(x[start : start + chunk], centres).argmin(axis=1)
    return out


class ProductQuantizer:
    """Jégou et al. (2011) PQ: split dims into ``m`` sub-spaces, 256-centroid codebook each."""

    def __init__(self, dim: int, m: int, ksub: int = 256) -> None:
        if dim % m != 0:
            raise ValueError("dim must be divisible by m")
        if not 2 <= ksub <= 256:
            raise ValueError("ksub must be in [2, 256]")
        self.dim, self.m, self.ksub = dim, m, ksub
        self.dsub = dim // m
        self.codebooks: F32 = np.zeros((m, ksub, self.dsub), dtype=np.float32)
        self.trained = False

    def train(self, x: F32, iters: int = 15, seed: int = 0) -> ProductQuantizer:
        if x.shape[0] < self.ksub:
            raise ValueError("need at least ksub training vectors")
        for j in range(self.m):
            sub = np.ascontiguousarray(x[:, j * self.dsub : (j + 1) * self.dsub])
            self.codebooks[j] = kmeans(sub, self.ksub, iters=iters, seed=seed + j)
        self.trained = True
        return self

    def encode(self, x: F32) -> U8:
        if not self.trained:
            raise RuntimeError("train the quantizer first")
        codes = np.empty((x.shape[0], self.m), dtype=np.uint8)
        for j in range(self.m):
            sub = x[:, j * self.dsub : (j + 1) * self.dsub]
            codes[:, j] = assign_nearest(sub, self.codebooks[j]).astype(np.uint8)
        return codes

    def decode(self, codes: U8) -> F32:
        parts = [self.codebooks[j][codes[:, j]] for j in range(self.m)]
        out: F32 = np.concatenate(parts, axis=1).astype(np.float32)
        return out

    def lut(self, query: F32) -> F32:
        """Asymmetric distance table: lut[j, c] = ||q_j - codebook[j, c]||^2, padded to 256."""
        table = np.full((self.m, 256), np.float32(np.inf), dtype=np.float32)
        for j in range(self.m):
            sub = query[j * self.dsub : (j + 1) * self.dsub].astype(np.float64)
            diff = self.codebooks[j].astype(np.float64) - sub
            table[j, : self.ksub] = (diff * diff).sum(axis=1).astype(np.float32)
        return table


@dataclass(slots=True)
class _InvList:
    ids: list[I64] = field(default_factory=list)
    codes: list[U8] = field(default_factory=list)


class IVFPQIndex:
    """Inverted file with residual product quantization and optional exact re-ranking."""

    def __init__(
        self,
        dim: int,
        nlist: int = 64,
        m: int = 8,
        metric: str = "l2",
        ksub: int = 256,
        bridge: KernelBridge | None = None,
    ) -> None:
        prepare(np.zeros((1, dim)), metric)
        self.dim, self.nlist, self.metric = dim, nlist, metric
        self.pq = ProductQuantizer(dim, m, ksub)
        self.bridge = bridge or get_bridge()
        self.centroids: F32 = np.zeros((nlist, dim), dtype=np.float32)
        self.lists = [_InvList() for _ in range(nlist)]
        self.raw: F32 = np.zeros((0, dim), dtype=np.float32)  # kept for optional re-ranking
        self.trained = False

    def train(self, x: npt.ArrayLike, seed: int = 0, max_train: int = 20_000) -> IVFPQIndex:
        arr = prepare(x, self.metric)
        if arr.shape[0] > max_train:
            pick = np.random.default_rng(seed).choice(arr.shape[0], max_train, replace=False)
            arr = arr[np.sort(pick)]
        self.centroids = kmeans(arr, self.nlist, seed=seed)
        residuals = arr - self.centroids[assign_nearest(arr, self.centroids)]
        self.pq.train(np.ascontiguousarray(residuals, dtype=np.float32), seed=seed)
        self.trained = True
        return self

    def add(self, x: npt.ArrayLike) -> I64:
        if not self.trained:
            raise RuntimeError("train the index first")
        arr = prepare(x, self.metric)
        start = self.raw.shape[0]
        ids = np.arange(start, start + arr.shape[0], dtype=np.int64)
        assign = assign_nearest(arr, self.centroids)
        codes = self.pq.encode(np.ascontiguousarray(arr - self.centroids[assign], dtype=np.float32))
        for lst in np.unique(assign):
            sel = assign == lst
            self.lists[int(lst)].ids.append(ids[sel])
            self.lists[int(lst)].codes.append(codes[sel])
        self.raw = np.ascontiguousarray(np.vstack([self.raw, arr]))
        return ids

    def __len__(self) -> int:
        return int(self.raw.shape[0])

    @property
    def code_bytes_per_vector(self) -> int:
        return self.pq.m

    def search(
        self, queries: npt.ArrayLike, k: int = 10, nprobe: int = 8, rerank: int = 0
    ) -> tuple[I64, F64]:
        """Probe the ``nprobe`` nearest lists with ADC; optionally re-rank ``rerank`` candidates exactly."""
        q = prepare(queries, self.metric)
        nprobe = min(nprobe, self.nlist)
        shortlist = max(k, rerank)
        out_ids = np.full((q.shape[0], k), -1, dtype=np.int64)
        out_d = np.full((q.shape[0], k), np.inf, dtype=np.float64)
        probe_d = sqdist(q, self.centroids)
        for qi in range(q.shape[0]):
            probes = np.argsort(probe_d[qi], kind="stable")[:nprobe]
            cand_ids: list[I64] = []
            cand_d: list[F64] = []
            for lst in probes:
                inv = self.lists[int(lst)]
                if not inv.ids:
                    continue
                ids = np.concatenate(inv.ids)
                codes = np.concatenate(inv.codes)
                lut = self.pq.lut(q[qi] - self.centroids[int(lst)])
                top_ids, top_d = self.bridge.pq_adc_topk(codes, lut, ids, shortlist)
                keep = top_ids >= 0
                cand_ids.append(top_ids[keep])
                cand_d.append(top_d[keep])
            if not cand_ids:
                continue
            all_ids = np.concatenate(cand_ids)
            all_d = np.concatenate(cand_d)
            order = np.lexsort((all_ids, all_d))[:shortlist]
            all_ids, all_d = all_ids[order], all_d[order]
            if rerank:
                all_d = self.bridge.l2_gather(q[qi], self.raw, all_ids)
                order = np.lexsort((all_ids, all_d))
                all_ids, all_d = all_ids[order], all_d[order]
            n = min(k, all_ids.shape[0])
            out_ids[qi, :n] = all_ids[:n]
            out_d[qi, :n] = all_d[:n]
        return out_ids, out_d
