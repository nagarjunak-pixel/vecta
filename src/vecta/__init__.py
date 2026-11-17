"""Vecta: vector search engine with HNSW, IVF-PQ, RaBitQ-style binary codes and hybrid retrieval."""

from vecta.datasets import Dataset, clustered_gaussians, recall_at_k
from vecta.graph import BinaryIndex, HNSWIndex
from vecta.hybrid import BM25, reciprocal_rank_fusion
from vecta.kernels import KernelBridge, get_bridge
from vecta.quantize import FlatIndex, IVFPQIndex, ProductQuantizer, kmeans

__version__ = "1.0.0"

__all__ = [
    "BM25",
    "BinaryIndex",
    "Dataset",
    "FlatIndex",
    "HNSWIndex",
    "IVFPQIndex",
    "KernelBridge",
    "ProductQuantizer",
    "__version__",
    "clustered_gaussians",
    "get_bridge",
    "kmeans",
    "recall_at_k",
    "reciprocal_rank_fusion",
]
