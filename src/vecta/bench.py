"""Reproducible ANN benchmark: recall@k / QPS / memory trade-offs and native-vs-NumPy kernels."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from vecta.datasets import Dataset, clustered_gaussians, recall_at_k
from vecta.graph import BinaryIndex, HNSWIndex
from vecta.kernels import KernelBridge
from vecta.quantize import FlatIndex, IVFPQIndex


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One operating point. ``distance_evals_per_query`` counts exact float32 distance
    evaluations (ADC lookups and Hamming popcounts are not counted)."""

    index: str
    params: str
    recall_at_10: float
    qps: float
    bytes_per_vector: int
    distance_evals_per_query: float


@dataclass(frozen=True, slots=True)
class KernelPoint:
    kernel: str
    native_ms: float
    numpy_ms: float
    speedup: float
    bit_exact: bool


def _timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def kernel_benchmarks(ds: Dataset, k: int = 10) -> list[KernelPoint]:
    native, numpy_b = KernelBridge(), KernelBridge(use_native=False)
    q = ds.queries[:20]
    points: list[KernelPoint] = []
    (n_ids, n_d), n_s = _timed(native.l2_topk, q, ds.base, k)
    (p_ids, p_d), p_s = _timed(numpy_b.l2_topk, q, ds.base, k)
    points.append(
        KernelPoint("exact L2 top-k (20 queries)", n_s * 1e3, p_s * 1e3, p_s / n_s,
                    bool(np.array_equal(n_ids, p_ids) and np.array_equal(n_d, p_d)))
    )  # fmt: skip
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 256, size=(ds.base.shape[0], 16), dtype=np.uint8)
    lut = rng.random((16, 256), dtype=np.float32)
    rows = np.arange(codes.shape[0], dtype=np.int64)
    (a_ids, a_d), a_s = _timed(native.pq_adc_topk, codes, lut, rows, k)
    (b_ids, b_d), b_s = _timed(numpy_b.pq_adc_topk, codes, lut, rows, k)
    points.append(
        KernelPoint("PQ ADC scan (m=16)", a_s * 1e3, b_s * 1e3, b_s / a_s,
                    bool(np.array_equal(a_ids, b_ids) and np.array_equal(a_d, b_d)))
    )  # fmt: skip
    bits = rng.integers(0, 2**63, size=(ds.base.shape[0], 2), dtype=np.uint64)
    query = bits[0].copy()
    (h_ids, h_d), h_s = _timed(native.hamming_topk, bits, query, k)
    (g_ids, g_d), g_s = _timed(numpy_b.hamming_topk, bits, query, k)
    points.append(
        KernelPoint("Hamming popcount top-k (128-bit)", h_s * 1e3, g_s * 1e3, g_s / h_s,
                    bool(np.array_equal(h_ids, g_ids) and np.array_equal(h_d, g_d)))
    )  # fmt: skip
    return points


def run_benchmark(
    n: int = 20_000,
    nq: int = 200,
    dim: int = 96,
    clusters: int = 64,
    spread: float = 0.6,
    seed: int = 0,
    hnsw_m: int = 16,
    ef_construction: int = 100,
) -> dict[str, Any]:
    ds = clustered_gaussians(n=n, nq=nq, dim=dim, clusters=clusters, spread=spread, seed=seed)
    k = 10
    flat = FlatIndex(dim)
    flat.add(ds.base)
    (truth, _), flat_s = _timed(flat.search, ds.queries, k)
    sweep: list[SweepPoint] = [SweepPoint("flat", "exact", 1.0, nq / flat_s, 4 * dim, float(n))]

    hnsw = HNSWIndex(dim, m=hnsw_m, ef_construction=ef_construction, seed=seed)
    _, hnsw_build = _timed(hnsw.add, ds.base)
    for ef in (16, 32, 64, 128):
        hnsw.distance_evals = 0
        (ids, _), s = _timed(hnsw.search, ds.queries, k, ef)
        sweep.append(
            SweepPoint("hnsw", f"M={hnsw_m},ef={ef}", recall_at_k(ids, truth, k), nq / s,
                       4 * dim + 4 * 2 * hnsw_m, hnsw.distance_evals / nq)
        )  # fmt: skip

    # Filtered search: only even ids are admissible.
    even = np.zeros(n, dtype=np.uint8)
    even[::2] = 1
    f_flat = FlatIndex(dim)
    f_flat.add(ds.base)
    f_flat.delete(np.flatnonzero(even == 0))
    f_truth, _ = f_flat.search(ds.queries, k)
    f_ids, _ = hnsw.search(ds.queries, k, ef=64, filter_fn=lambda i: i % 2 == 0)
    filtered_recall = recall_at_k(f_ids, f_truth, k)

    m = 16
    ivf = IVFPQIndex(dim, nlist=max(16, int(np.sqrt(n))), m=m)
    _, ivf_build = _timed(lambda: ivf.train(ds.base, seed=seed).add(ds.base))
    for nprobe, rerank in ((4, 0), (16, 0), (8, 50), (16, 100)):
        (ids, _), s = _timed(ivf.search, ds.queries, k, nprobe, rerank)
        sweep.append(
            SweepPoint("ivf-pq", f"nlist={ivf.nlist},m={m},nprobe={nprobe},rerank={rerank}",
                       recall_at_k(ids, truth, k), nq / s, m, float(rerank))
        )  # fmt: skip

    binary = BinaryIndex(dim, seed=seed)
    _, bin_build = _timed(lambda: binary.train(ds.base).add(ds.base))
    for rerank in (50, 200, 500):
        (ids, _), s = _timed(binary.search, ds.queries, k, rerank)
        sweep.append(
            SweepPoint("binary+rerank", f"bits={dim},rerank={rerank}", recall_at_k(ids, truth, k),
                       nq / s, binary.code_bytes_per_vector, float(rerank))
        )  # fmt: skip

    return {
        "dataset": {"name": ds.name, "n": n, "nq": nq, "dim": dim, "clusters": clusters, "spread": spread},
        "build_seconds": {"hnsw": hnsw_build, "ivf_pq": ivf_build, "binary": bin_build},
        "hnsw_graph": hnsw.graph_stats(),
        "hnsw_filtered_even_ids_recall_at_10_ef64": filtered_recall,
        "sweep": [asdict(p) for p in sweep],
        "kernels": [asdict(p) for p in kernel_benchmarks(ds, k)],
        "native_backend": KernelBridge().backend,
    }
