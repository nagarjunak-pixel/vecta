"""Deterministic synthetic embedding datasets and retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Dataset:
    base: F32
    queries: F32
    name: str


def clustered_gaussians(
    n: int = 10_000,
    nq: int = 100,
    dim: int = 64,
    clusters: int = 32,
    spread: float = 0.35,
    seed: int = 0,
) -> Dataset:
    """Mixture of anisotropic Gaussians on a low-dimensional manifold.

    Real sentence embeddings are clustered and have a decaying spectrum, so the generator
    draws cluster centres, then scales each axis by a power-law spectrum. Queries come from the
    same mixture (held-out draws), as in ANN-Benchmarks style evaluation.
    """
    rng = np.random.default_rng(seed)
    spectrum = (1.0 / np.arange(1, dim + 1) ** 0.5).astype(np.float64)
    centres = rng.normal(size=(clusters, dim)) * spectrum

    def draw(count: int) -> F32:
        assign = rng.integers(0, clusters, size=count)
        noise = rng.normal(size=(count, dim)) * spectrum * spread
        out: F32 = (centres[assign] + noise).astype(np.float32)
        return out

    return Dataset(base=draw(n), queries=draw(nq), name=f"gauss-{clusters}c-{dim}d-{n}")


def recall_at_k(found: I64, truth: I64, k: int) -> float:
    """Mean |found[:k] ∩ truth[:k]| / k over queries (ANN-Benchmarks definition)."""
    if found.shape[0] != truth.shape[0]:
        raise ValueError("found and truth must have the same number of queries")
    hits = 0
    for f, t in zip(found[:, :k], truth[:, :k], strict=True):
        hits += len(set(f.tolist()) & set(t.tolist()) - {-1})
    return float(hits) / float(k * int(found.shape[0]))
