"""SHA-256-cached C++20 kernel bridge (ctypes) with bit-exact NumPy twins.

Both paths accumulate squared differences / LUT entries sequentially in float64 over float32
inputs, the C++ build disables FP contraction (``-ffp-contract=off``), and every top-k orders
by ``(distance, id)``. The native and Python results are therefore identical, not merely close.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
U8 = npt.NDArray[np.uint8]
U64 = npt.NDArray[np.uint64]

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = _REPO_ROOT / "csrc" / "vecta_kernels.cpp"
COMPILE_FLAGS = ["-O3", "-std=c++20", "-ffp-contract=off", "-shared", "-fPIC"]

_f32p = ctypes.POINTER(ctypes.c_float)
_f64p = ctypes.POINTER(ctypes.c_double)
_i64p = ctypes.POINTER(ctypes.c_int64)
_u8p = ctypes.POINTER(ctypes.c_uint8)
_u64p = ctypes.POINTER(ctypes.c_uint64)


def _topk_rows(dists: F64, ids: I64, k: int) -> tuple[I64, F64]:
    """Top-k by (distance, id) with -1 / +inf padding, matching the C++ heap semantics."""
    out_ids = np.full(k, -1, dtype=np.int64)
    out_d = np.full(k, np.inf, dtype=np.float64)
    if dists.size:
        order = np.lexsort((ids, dists))[:k]
        out_ids[: order.size] = ids[order]
        out_d[: order.size] = dists[order]
    return out_ids, out_d


def py_l2_rows(query: F32, base: F32) -> F64:
    """Sequential float64 accumulation of squared differences (one query vs many rows)."""
    acc = np.zeros(base.shape[0], dtype=np.float64)
    q64 = query.astype(np.float64)
    for d in range(base.shape[1]):
        diff = q64[d] - base[:, d].astype(np.float64)
        acc += diff * diff
    return acc


class KernelBridge:
    """Loads ``csrc/vecta_kernels.cpp``; ``use_native=False`` or ``VECTA_DISABLE_NATIVE=1`` forces NumPy."""

    def __init__(self, source: Path | None = None, use_native: bool | None = None) -> None:
        self._src = source or DEFAULT_SOURCE
        self._lib: ctypes.CDLL | None = None
        self.backend = "numpy-fallback"
        if use_native is None:
            use_native = os.environ.get("VECTA_DISABLE_NATIVE", "0") != "1"
        if use_native:
            self._load()

    @property
    def is_native(self) -> bool:
        return self._lib is not None

    def _load(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None or not self._src.exists():
            return
        cache = self._src.parent.parent / ".vecta_cache"
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(self._src.read_bytes() + " ".join(COMPILE_FLAGS).encode()).hexdigest()[:12]
        ext = "dylib" if platform.system() == "Darwin" else "so"
        lib_path = cache / f"libvecta_{digest}.{ext}"
        if not lib_path.exists():
            tmp = cache / f"libvecta_{digest}.{os.getpid()}.tmp"
            proc = subprocess.run(
                [compiler, *COMPILE_FLAGS, str(self._src), "-o", str(tmp)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return
            tmp.replace(lib_path)
        lib = ctypes.CDLL(str(lib_path))
        lib.vecta_l2_topk.argtypes = [
            _f32p, ctypes.c_int64, _f32p, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
            _u8p, _i64p, _f64p,
        ]  # fmt: skip
        lib.vecta_l2_topk.restype = None
        lib.vecta_l2_gather.argtypes = [_f32p, _f32p, ctypes.c_int32, _i64p, ctypes.c_int64, _f64p]
        lib.vecta_l2_gather.restype = None
        lib.vecta_pq_adc_topk.argtypes = [
            _u8p, ctypes.c_int64, ctypes.c_int32, _f32p, _i64p, ctypes.c_int32, _i64p, _f64p,
        ]  # fmt: skip
        lib.vecta_pq_adc_topk.restype = None
        lib.vecta_hamming_topk.argtypes = [
            _u64p, ctypes.c_int64, ctypes.c_int32, _u64p, _u8p, ctypes.c_int32, _i64p, _f64p,
        ]  # fmt: skip
        lib.vecta_hamming_topk.restype = None
        self._lib = lib
        self.backend = f"c++20-native ({Path(compiler).name} -O3)"

    # ------------------------------------------------------------------------------ exact L2

    def l2_topk(self, queries: F32, base: F32, k: int, alive: U8 | None = None) -> tuple[I64, F64]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        b = np.ascontiguousarray(base, dtype=np.float32)
        nq, dim = q.shape
        ids = np.empty((nq, k), dtype=np.int64)
        dists = np.empty((nq, k), dtype=np.float64)
        if self._lib is not None:
            alive_c = None if alive is None else np.ascontiguousarray(alive, dtype=np.uint8)
            self._lib.vecta_l2_topk(
                q.ctypes.data_as(_f32p), nq, b.ctypes.data_as(_f32p), b.shape[0], dim, k,
                None if alive_c is None else alive_c.ctypes.data_as(_u8p),
                ids.ctypes.data_as(_i64p), dists.ctypes.data_as(_f64p),
            )  # fmt: skip
            return ids, dists
        rows = np.arange(b.shape[0], dtype=np.int64)
        mask = np.ones(b.shape[0], dtype=bool) if alive is None else alive.astype(bool)
        for i in range(nq):
            d = py_l2_rows(q[i], b[mask])
            ids[i], dists[i] = _topk_rows(d, rows[mask], k)
        return ids, dists

    def l2_gather(self, query: F32, base: F32, ids: I64) -> F64:
        idx = np.ascontiguousarray(ids, dtype=np.int64)
        if self._lib is not None:
            out = np.empty(idx.shape[0], dtype=np.float64)
            self._lib.vecta_l2_gather(
                query.ctypes.data_as(_f32p), base.ctypes.data_as(_f32p), base.shape[1],
                idx.ctypes.data_as(_i64p), idx.shape[0], out.ctypes.data_as(_f64p),
            )  # fmt: skip
            return out
        return py_l2_rows(query, base[idx])

    # -------------------------------------------------------------------------------- PQ ADC

    def pq_adc_topk(self, codes: U8, lut: F32, row_ids: I64, k: int) -> tuple[I64, F64]:
        c = np.ascontiguousarray(codes, dtype=np.uint8)
        table = np.ascontiguousarray(lut, dtype=np.float32)
        rid = np.ascontiguousarray(row_ids, dtype=np.int64)
        n, m = c.shape
        if self._lib is not None:
            ids = np.empty(k, dtype=np.int64)
            dists = np.empty(k, dtype=np.float64)
            self._lib.vecta_pq_adc_topk(
                c.ctypes.data_as(_u8p), n, m, table.ctypes.data_as(_f32p), rid.ctypes.data_as(_i64p),
                k, ids.ctypes.data_as(_i64p), dists.ctypes.data_as(_f64p),
            )  # fmt: skip
            return ids, dists
        acc = np.zeros(n, dtype=np.float64)
        for j in range(m):
            acc += table[j, c[:, j]].astype(np.float64)
        return _topk_rows(acc, rid, k)

    # ------------------------------------------------------------------------------- Hamming

    def hamming_topk(self, codes: U64, query: U64, k: int, alive: U8 | None = None) -> tuple[I64, F64]:
        c = np.ascontiguousarray(codes, dtype=np.uint64)
        qc = np.ascontiguousarray(query, dtype=np.uint64)
        n, words = c.shape
        if self._lib is not None:
            ids = np.empty(k, dtype=np.int64)
            dists = np.empty(k, dtype=np.float64)
            alive_c = None if alive is None else np.ascontiguousarray(alive, dtype=np.uint8)
            self._lib.vecta_hamming_topk(
                c.ctypes.data_as(_u64p), n, words, qc.ctypes.data_as(_u64p),
                None if alive_c is None else alive_c.ctypes.data_as(_u8p),
                k, ids.ctypes.data_as(_i64p), dists.ctypes.data_as(_f64p),
            )  # fmt: skip
            return ids, dists
        xor = np.bitwise_xor(c, qc[None, :])
        dist = np.unpackbits(xor.view(np.uint8), axis=1).sum(axis=1).astype(np.float64)
        rows = np.arange(n, dtype=np.int64)
        mask = np.ones(n, dtype=bool) if alive is None else alive.astype(bool)
        return _topk_rows(dist[mask], rows[mask], k)


_BRIDGE: KernelBridge | None = None


def get_bridge() -> KernelBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = KernelBridge()
    return _BRIDGE
