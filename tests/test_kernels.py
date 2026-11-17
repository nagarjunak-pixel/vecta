from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vecta.kernels import DEFAULT_SOURCE, KernelBridge, get_bridge

NATIVE = KernelBridge(use_native=True)
NUMPY = KernelBridge(use_native=False)


def _exact_truth(q: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    d = ((q[:, None, :].astype(np.float64) - b[None, :, :].astype(np.float64)) ** 2).sum(axis=2)
    ids = np.arange(b.shape[0])
    return np.stack([ids[np.lexsort((ids, row))][:k] for row in d])


def test_native_backend_is_built() -> None:
    assert NATIVE.is_native, "clang++/g++ is required to build the C++20 kernels"
    assert NATIVE.backend.startswith("c++20-native")
    assert NUMPY.backend == "numpy-fallback"
    assert get_bridge() is get_bridge()


def test_l2_topk_bit_exact_and_correct() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 24)).astype(np.float32)
    q = rng.normal(size=(7, 24)).astype(np.float32)
    n_ids, n_d = NATIVE.l2_topk(q, base, 10)
    p_ids, p_d = NUMPY.l2_topk(q, base, 10)
    assert np.array_equal(n_ids, p_ids) and np.array_equal(n_d, p_d)
    assert np.array_equal(n_ids, _exact_truth(q, base, 10))
    assert np.all(np.diff(n_d, axis=1) >= 0)


def test_l2_topk_alive_mask_and_padding() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(6, 4)).astype(np.float32)
    alive = np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8)
    for bridge in (NATIVE, NUMPY):
        ids, d = bridge.l2_topk(base[:1], base, 5, alive)
        assert set(ids[0, :3].tolist()) == {0, 2, 4}
        assert ids[0, 3:].tolist() == [-1, -1]
        assert np.isinf(d[0, 3:]).all()
        assert ids[0, 0] == 0 and d[0, 0] == 0.0


def test_l2_gather_equal() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(size=(50, 16)).astype(np.float32)
    ids = np.array([3, 3, 0, 49], dtype=np.int64)
    a = NATIVE.l2_gather(base[5], base, ids)
    b = NUMPY.l2_gather(base[5], base, ids)
    assert np.array_equal(a, b)
    assert np.allclose(a, ((base[ids] - base[5]) ** 2).sum(axis=1), rtol=1e-5)


def test_pq_adc_topk_equal_and_padded() -> None:
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 256, size=(300, 8), dtype=np.uint8)
    lut = rng.random((8, 256), dtype=np.float32)
    rows = np.arange(1000, 1300, dtype=np.int64)
    a = NATIVE.pq_adc_topk(codes, lut, rows, 12)
    b = NUMPY.pq_adc_topk(codes, lut, rows, 12)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    brute = np.array([sum(float(lut[j, c[j]]) for j in range(8)) for c in codes])
    assert a[0][0] == rows[int(np.lexsort((rows, brute))[0])]
    few = NATIVE.pq_adc_topk(codes[:2], lut, rows[:2], 4)
    assert few[0][2:].tolist() == [-1, -1]
    assert NUMPY.pq_adc_topk(codes[:0], lut, rows[:0], 2)[0].tolist() == [-1, -1]


def test_hamming_topk_equal_with_mask() -> None:
    rng = np.random.default_rng(4)
    codes = rng.integers(0, 2**63, size=(200, 3), dtype=np.uint64)
    query = codes[17].copy()
    alive = np.ones(200, dtype=np.uint8)
    alive[17] = 0
    a = NATIVE.hamming_topk(codes, query, 5)
    b = NUMPY.hamming_topk(codes, query, 5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[0][0] == 17 and a[1][0] == 0.0
    c = NATIVE.hamming_topk(codes, query, 5, alive)
    d = NUMPY.hamming_topk(codes, query, 5, alive)
    assert np.array_equal(c[0], d[0]) and 17 not in c[0].tolist()
    brute = np.array(
        [sum(bin(int(x) ^ int(y)).count("1") for x, y in zip(row, query, strict=True)) for row in codes]
    )
    assert c[1][0] == np.sort(brute[alive == 1])[0]


@given(
    n=st.integers(1, 40),
    dim=st.integers(1, 9),
    k=st.integers(1, 12),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=60, deadline=None)
def test_property_native_equals_numpy(n: int, dim: int, k: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, dim)).astype(np.float32)
    base[rng.integers(0, n, size=n // 3)] = base[0]  # duplicates force (distance, id) tie-breaks
    q = rng.normal(size=(2, dim)).astype(np.float32)
    a, b = NATIVE.l2_topk(q, base, k), NUMPY.l2_topk(q, base, k)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    codes = rng.integers(0, 4, size=(n, 3), dtype=np.uint8)  # many exact ADC ties
    lut = rng.random((3, 256), dtype=np.float32)
    rows = np.arange(n, dtype=np.int64)
    c, d = NATIVE.pq_adc_topk(codes, lut, rows, k), NUMPY.pq_adc_topk(codes, lut, rows, k)
    assert np.array_equal(c[0], d[0]) and np.array_equal(c[1], d[1])


def test_bridge_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not KernelBridge(source=tmp_path / "missing.cpp").is_native
    broken = tmp_path / "csrc" / "broken.cpp"
    broken.parent.mkdir()
    broken.write_text("not c++ at all")
    assert not KernelBridge(source=broken).is_native
    fresh = tmp_path / "csrc" / "fresh.cpp"
    fresh.write_text(DEFAULT_SOURCE.read_text())
    assert KernelBridge(source=fresh).is_native  # compiles into a fresh cache
    monkeypatch.setenv("VECTA_DISABLE_NATIVE", "1")
    assert not KernelBridge().is_native
    monkeypatch.setenv("VECTA_DISABLE_NATIVE", "0")
    monkeypatch.setenv("PATH", "")
    assert not KernelBridge().is_native
    assert os.environ["PATH"] == ""
