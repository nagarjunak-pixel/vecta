"""FastAPI service: named in-memory collections with filtered vector search and hybrid BM25 + RRF."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vecta import __version__
from vecta.graph import HNSWIndex
from vecta.hybrid import BM25, reciprocal_rank_fusion
from vecta.quantize import FlatIndex


class CreateCollection(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    dim: int = Field(ge=1, le=4096)
    index: Literal["hnsw", "flat"] = "hnsw"
    metric: Literal["l2", "cosine"] = "cosine"
    m: int = Field(default=16, ge=2, le=128)
    ef_construction: int = Field(default=100, ge=4, le=2000)


class AddPoints(BaseModel):
    vectors: list[list[float]] = Field(min_length=1)
    texts: list[str] | None = None
    metadata: list[dict[str, Any]] | None = None


class SearchRequest(BaseModel):
    vector: list[float]
    k: int = Field(default=10, ge=1, le=1000)
    ef: int = Field(default=64, ge=1, le=4096)
    filter: dict[str, Any] | None = None


class HybridRequest(SearchRequest):
    text: str = Field(min_length=1)
    rrf_c: float = Field(default=60.0, gt=0)


class Hit(BaseModel):
    id: int
    score: float
    text: str | None
    metadata: dict[str, Any]


@dataclass
class Collection:
    spec: CreateCollection
    index: HNSWIndex | FlatIndex
    texts: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    deleted: set[int] = field(default_factory=set)

    def matches(self, i: int, flt: dict[str, Any] | None) -> bool:
        if i in self.deleted:
            return False
        return flt is None or all(self.metadata[i].get(k) == v for k, v in flt.items())

    def vector_search(self, req: SearchRequest) -> list[tuple[int, float]]:
        if len(req.vector) != self.spec.dim:
            raise HTTPException(status_code=422, detail=f"vector must have dim {self.spec.dim}")
        q = np.asarray([req.vector], dtype=np.float32)
        if isinstance(self.index, HNSWIndex):
            flt = req.filter
            ids, d = self.index.search(q, req.k, ef=req.ef, filter_fn=lambda i: self.matches(i, flt))
        else:
            n = self.index.vectors.shape[0]
            ids, d = self.index.search(q, n)
            keep = [j for j in range(n) if ids[0, j] >= 0 and self.matches(int(ids[0, j]), req.filter)]
            ids, d = ids[:, keep[: req.k]], d[:, keep[: req.k]]
        return [(int(i), float(x)) for i, x in zip(ids[0], d[0], strict=True) if i >= 0]

    def hit(self, i: int, score: float) -> Hit:
        return Hit(id=i, score=score, text=self.texts[i] or None, metadata=self.metadata[i])


def create_app() -> FastAPI:
    app = FastAPI(title="Vecta", version=__version__)
    collections: dict[str, Collection] = {}

    def get(name: str) -> Collection:
        if name not in collections:
            raise HTTPException(status_code=404, detail=f"collection {name!r} not found")
        return collections[name]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/collections", status_code=201)
    def create(spec: CreateCollection) -> dict[str, Any]:
        if spec.name in collections:
            raise HTTPException(status_code=409, detail="collection already exists")
        index: HNSWIndex | FlatIndex
        if spec.index == "hnsw":
            index = HNSWIndex(spec.dim, m=spec.m, ef_construction=spec.ef_construction, metric=spec.metric)
        else:
            index = FlatIndex(spec.dim, metric=spec.metric)
        collections[spec.name] = Collection(spec=spec, index=index)
        return spec.model_dump()

    @app.get("/v1/collections")
    def list_collections() -> list[dict[str, Any]]:
        return [
            {"name": n, "size": len(c.index) - len(c.deleted), "index": c.spec.index}
            for n, c in collections.items()
        ]

    @app.post("/v1/collections/{name}/points")
    def add(name: str, body: AddPoints) -> dict[str, list[int]]:
        col = get(name)
        n = len(body.vectors)
        if any(len(v) != col.spec.dim for v in body.vectors):
            raise HTTPException(status_code=422, detail=f"all vectors must have dim {col.spec.dim}")
        for extra, label in ((body.texts, "texts"), (body.metadata, "metadata")):
            if extra is not None and len(extra) != n:
                raise HTTPException(status_code=422, detail=f"{label} must match the number of vectors")
        ids = col.index.add(np.asarray(body.vectors, dtype=np.float32))
        col.texts.extend(body.texts or [""] * n)
        col.metadata.extend(body.metadata or [{} for _ in range(n)])
        return {"ids": ids.tolist()}

    @app.delete("/v1/collections/{name}/points/{point_id}")
    def delete(name: str, point_id: int) -> dict[str, int]:
        col = get(name)
        if not 0 <= point_id < len(col.metadata):
            raise HTTPException(status_code=404, detail="point not found")
        col.deleted.add(point_id)
        return {"deleted": point_id}

    @app.post("/v1/collections/{name}/search", response_model=list[Hit])
    def search(name: str, req: SearchRequest) -> list[Hit]:
        col = get(name)
        return [col.hit(i, d) for i, d in col.vector_search(req)]

    @app.post("/v1/collections/{name}/hybrid", response_model=list[Hit])
    def hybrid(name: str, req: HybridRequest) -> list[Hit]:
        col = get(name)
        if not any(col.texts):
            raise HTTPException(status_code=422, detail="collection has no texts for BM25")
        dense = [i for i, _ in col.vector_search(req.model_copy(update={"k": max(req.k, 50)}))]
        lexical = [i for i in BM25(col.texts).search(req.text, max(req.k, 50)) if col.matches(i, req.filter)]
        fused = reciprocal_rank_fusion([dense, lexical], k=req.k, c=req.rrf_c)
        return [col.hit(i, s) for i, s in fused]

    return app
