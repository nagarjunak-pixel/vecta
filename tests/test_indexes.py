from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vecta.datasets import clustered_gaussians, recall_at_k
from vecta.graph import BinaryIndex, HNSWIndex
from vecta.hybrid import BM25, reciprocal_rank_fusion, tokenize
from vecta.kernels import KernelBridge
from vecta.quantize import FlatIndex, IVFPQIndex, ProductQuantizer, kmeans, prepare, sqdist

DS = clustered_gaussians(n=2000, nq=40, dim=32, clusters=16, spread=0.6, seed=3)
FLAT = FlatIndex(32)
FLAT.add(DS.base)
TRUTH, _ = FLAT.search(DS.queries, 10)


@pytest.fixture(scope="module")
def hnsw() -> HNSWIndex:
    idx = HNSWIndex(32, m=8, ef_construction=64, seed=1)
    idx.add(DS.base)
    return idx


# --------------------------------------------------------------------------- flat & prepare


def test_prepare_and_metrics() -> None:
    with pytest.raises(ValueError):
        prepare([[1.0]], "manhattan")
    unit = prepare([[3.0, 4.0], [0.0, 0.0]], "cosine")
    assert np.allclose(unit[0], [0.6, 0.8]) and np.array_equal(unit[1], [0.0, 0.0])
    assert prepare([1.0, 2.0], "l2").shape == (1, 2)


def test_flat_index_add_search_delete() -> None:
    idx = FlatIndex(3)
    ids = idx.add([[0, 0, 0], [1, 0, 0], [5, 5, 5]])
    assert ids.tolist() == [0, 1, 2] and len(idx) == 3
    found, d = idx.search([[0.9, 0, 0]], 2)
    assert found[0].tolist() == [1, 0] and d[0, 0] == pytest.approx(0.01, rel=1e-4)
    idx.delete([1])
    assert idx.search([[0.9, 0, 0]], 1)[0][0, 0] == 0 and len(idx) == 2
    with pytest.raises(ValueError):
        idx.add([[1.0, 2.0]])


def test_flat_cosine_ranking() -> None:
    idx = FlatIndex(2, metric="cosine")
    idx.add([[10.0, 0.0], [1.0, 1.0], [0.0, 3.0]])
    assert idx.search([[2.0, 0.1]], 3)[0][0].tolist() == [0, 1, 2]


def test_flat_native_and_numpy_agree() -> None:
    a = FlatIndex(32, bridge=KernelBridge(use_native=True))
    b = FlatIndex(32, bridge=KernelBridge(use_native=False))
    a.add(DS.base[:300])
    b.add(DS.base[:300])
    ra, rb = a.search(DS.queries, 5), b.search(DS.queries, 5)
    assert np.array_equal(ra[0], rb[0]) and np.array_equal(ra[1], rb[1])


# ------------------------------------------------------------------------- k-means & PQ


def test_sqdist_matches_direct() -> None:
    x = DS.base[:20]
    c = DS.base[20:25]
    direct = ((x[:, None, :].astype(np.float64) - c[None].astype(np.float64)) ** 2).sum(axis=2)
    assert np.allclose(sqdist(x, c), direct, atol=1e-6)


def test_kmeans_recovers_separated_clusters_and_edge_cases() -> None:
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 10], [-10, 10]], dtype=np.float32)
    x = np.concatenate([c + rng.normal(scale=0.1, size=(50, 2)) for c in centres]).astype(np.float32)
    found = kmeans(x, 3, seed=1)
    for c in centres:
        assert np.min(np.linalg.norm(found - c, axis=1)) < 0.2
    same = np.ones((10, 2), dtype=np.float32)
    assert kmeans(same, 3).shape == (3, 2)  # all-identical points exercise empty-cluster reseeding
    with pytest.raises(ValueError):
        kmeans(x, 0)
    with pytest.raises(ValueError):
        kmeans(x[:2], 3)


def test_product_quantizer_roundtrip_and_lut() -> None:
    with pytest.raises(ValueError):
        ProductQuantizer(30, 4)
    with pytest.raises(ValueError):
        ProductQuantizer(32, 4, ksub=300)
    pq = ProductQuantizer(32, 4, ksub=64)
    with pytest.raises(RuntimeError):
        pq.encode(DS.base[:5])
    with pytest.raises(ValueError):
        pq.train(DS.base[:10])
    pq.train(DS.base[:1000])
    codes = pq.encode(DS.base[:200])
    recon = pq.decode(codes)
    rel_err = np.linalg.norm(recon - DS.base[:200]) / np.linalg.norm(DS.base[:200])
    assert rel_err < 0.5
    lut = pq.lut(DS.queries[0])
    adc = np.array([sum(float(lut[j, c[j]]) for j in range(4)) for c in codes])
    direct = ((recon.astype(np.float64) - DS.queries[0].astype(np.float64)) ** 2).sum(axis=1)
    assert np.allclose(adc, direct, rtol=1e-4)
    assert np.isinf(lut[:, 64:]).all()


def test_ivfpq_recall_rerank_and_edge_cases() -> None:
    ivf = IVFPQIndex(32, nlist=16, m=8, ksub=64)
    with pytest.raises(RuntimeError):
        ivf.add(DS.base)
    ivf.train(DS.base, seed=0, max_train=1500)
    ivf.add(DS.base)
    assert len(ivf) == 2000 and ivf.code_bytes_per_vector == 8
    coarse, _ = ivf.search(DS.queries, 10, nprobe=4)
    fine, dists = ivf.search(DS.queries, 10, nprobe=8, rerank=100)
    assert recall_at_k(fine, TRUTH, 10) >= 0.95
    assert recall_at_k(fine, TRUTH, 10) >= recall_at_k(coarse, TRUTH, 10)
    assert np.all(np.diff(dists, axis=1) >= 0)
    empty = IVFPQIndex(32, nlist=4, m=8, ksub=16).train(DS.base[:200])
    ids, d = empty.search(DS.queries[:2], 3)
    assert (ids == -1).all() and np.isinf(d).all()
    sparse = IVFPQIndex(32, nlist=8, m=8, ksub=16).train(DS.base[:400])
    sparse.add(DS.base[:3])  # most inverted lists stay empty
    assert (sparse.search(DS.queries[:1], 3, nprobe=8)[0] >= 0).sum() == 3


# ----------------------------------------------------------------------------------- HNSW


def test_hnsw_recall_and_graph_invariants(hnsw: HNSWIndex) -> None:
    ids, d = hnsw.search(DS.queries, 10, ef=64)
    assert recall_at_k(ids, TRUTH, 10) >= 0.95
    assert np.all(np.diff(d, axis=1) >= 0)
    stats = hnsw.graph_stats()
    assert stats["reachable_fraction"] == 1.0 and stats["nodes"] == 2000
    for node, node_links in enumerate(hnsw.links):
        assert len(node_links[0]) <= hnsw.m_max0
        assert all(len(layer) <= hnsw.m for layer in node_links[1:])
        assert node not in node_links[0]
    low = recall_at_k(hnsw.search(DS.queries, 10, ef=10)[0], TRUTH, 10)
    assert low <= recall_at_k(ids, TRUTH, 10)
    assert len(hnsw) == 2000 and hnsw.vectors.shape == (2000, 32)


def test_hnsw_filter_and_delete() -> None:
    idx = HNSWIndex(32, m=8, ef_construction=48, seed=2)
    idx.add(DS.base[:800])
    flt, _ = idx.search(DS.queries[:10], 5, ef=64, filter_fn=lambda i: i % 3 == 0)
    assert all(i % 3 == 0 for i in flt.ravel().tolist())
    top = int(idx.search(DS.queries[:1], 1)[0][0, 0])
    idx.delete([top])
    again, _ = idx.search(DS.queries[:1], 5)
    assert top not in again[0].tolist() and len(idx) == 799
    with pytest.raises(IndexError):
        idx.delete([10_000])


def test_hnsw_errors_empty_and_cosine(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        HNSWIndex(8, m=1)
    idx = HNSWIndex(4, metric="cosine")
    ids, d = idx.search([[1, 0, 0, 0]], 3)
    assert (ids == -1).all() and np.isinf(d).all()
    with pytest.raises(ValueError):
        idx.add([[1.0, 2.0]])
    idx.add([[1, 0, 0, 0], [0, 1, 0, 0], [10, 1, 0, 0]])
    assert idx.search([[5, 0, 0, 0]], 1)[0][0, 0] == 0
    empty = HNSWIndex(4)
    empty.save(tmp_path / "empty.npz")
    restored = HNSWIndex.load(tmp_path / "empty.npz")
    assert restored.count == 0 and restored.entry == -1
    assert empty.graph_stats()["reachable_fraction"] == 1.0


def test_hnsw_save_load_roundtrip(hnsw: HNSWIndex, tmp_path: Path) -> None:
    path = tmp_path / "graph.npz"
    hnsw.delete([5])
    hnsw.save(path)
    restored = HNSWIndex.load(path)
    a = hnsw.search(DS.queries[:10], 10, ef=40)
    b = restored.search(DS.queries[:10], 10, ef=40)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert restored.deleted == {5} and restored.links == hnsw.links
    restored.add(DS.queries[:3])  # capacity grows after load
    assert restored.count == 2003
    hnsw.deleted.clear()


# --------------------------------------------------------------------------------- binary


def test_binary_index_recall_and_codes() -> None:
    idx = BinaryIndex(32, seed=0)
    with pytest.raises(RuntimeError):
        idx.add(DS.base)
    idx.train(DS.base).add(DS.base)
    assert len(idx) == 2000 and idx.code_bytes_per_vector == 8
    assert np.allclose(idx.rotation @ idx.rotation.T, np.eye(32), atol=1e-5)
    ids, d = idx.search(DS.queries, 10, rerank=300)
    assert recall_at_k(ids, TRUTH, 10) >= 0.9
    assert np.all(np.diff(d, axis=1) >= 0)
    bits = np.unpackbits(idx.codes[:1].view(np.uint8), bitorder="little")[:32]
    proj = (DS.base[0] - idx.mean) @ idx.rotation.T
    assert np.array_equal(bits.astype(bool), proj > 0)
    tiny = BinaryIndex(70)
    tiny.train(np.eye(70, dtype=np.float32)).add(np.eye(70, dtype=np.float32)[:3])
    assert tiny.words == 2 and tiny.search(np.eye(70, dtype=np.float32)[:1], 5)[0][0, 3:].tolist() == [-1, -1]


# --------------------------------------------------------------------------------- hybrid


def test_bm25_and_rrf() -> None:
    docs = [
        "HNSW graphs give logarithmic search",
        "product quantization compresses vectors",
        "graphs graphs graphs everywhere",
        "binary codes and hamming distance",
    ]
    bm = BM25(docs)
    assert bm.search("hnsw graph search")[0] == 0
    assert bm.search("quantization")[0] == 1
    assert bm.search("nothing matches") == []
    assert tokenize("Hello, World-42!") == ["hello", "world", "42"]
    with pytest.raises(ValueError):
        BM25([])
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, -1]], k=3, c=60)
    assert [d for d, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_dataset_determinism_and_recall_metric() -> None:
    a = clustered_gaussians(n=50, nq=5, dim=8, seed=9)
    b = clustered_gaussians(n=50, nq=5, dim=8, seed=9)
    assert np.array_equal(a.base, b.base) and a.base.dtype == np.float32
    truth = np.array([[1, 2, 3]])
    assert recall_at_k(np.array([[3, 9, -1]]), truth, 3) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((2, 3), dtype=np.int64), truth, 3)
