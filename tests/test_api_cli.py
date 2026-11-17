from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from vecta import __version__
from vecta.api import create_app
from vecta.cli import app as cli_app

VECS = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
TEXTS = ["hnsw graph index", "graph search engine", "product quantization", "bm25 lexical retrieval"]
META = [{"lang": "en"}, {"lang": "de"}, {"lang": "en"}, {"lang": "en"}]


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.parametrize("index", ["hnsw", "flat"])
def test_collection_lifecycle(client: TestClient, index: str) -> None:
    assert client.get("/healthz").json()["version"] == __version__
    r = client.post("/v1/collections", json={"name": "docs", "dim": 3, "index": index, "m": 4})
    assert r.status_code == 201
    assert client.post("/v1/collections", json={"name": "docs", "dim": 3}).status_code == 409
    ids = client.post(
        "/v1/collections/docs/points", json={"vectors": VECS, "texts": TEXTS, "metadata": META}
    ).json()
    assert ids == {"ids": [0, 1, 2, 3]}
    hits = client.post("/v1/collections/docs/search", json={"vector": [1, 0.05, 0], "k": 2}).json()
    assert [h["id"] for h in hits] == [0, 1]
    assert hits[0]["text"] == TEXTS[0] and hits[0]["metadata"] == {"lang": "en"}
    filtered = client.post(
        "/v1/collections/docs/search", json={"vector": [1, 0.05, 0], "k": 2, "filter": {"lang": "de"}}
    ).json()
    assert [h["id"] for h in filtered] == [1]
    assert client.delete("/v1/collections/docs/points/0").json() == {"deleted": 0}
    hits = client.post("/v1/collections/docs/search", json={"vector": [1, 0, 0], "k": 1}).json()
    assert hits[0]["id"] == 1
    fused = client.post(
        "/v1/collections/docs/hybrid", json={"vector": [0, 0, 1], "text": "graph search", "k": 3}
    ).json()
    assert {h["id"] for h in fused} >= {1, 3}
    assert all(h["id"] != 0 for h in fused)
    listing = client.get("/v1/collections").json()
    assert listing == [{"name": "docs", "size": 3, "index": index}]


def test_api_validation_errors(client: TestClient) -> None:
    assert client.post("/v1/collections/nope/search", json={"vector": [1]}).status_code == 404
    client.post("/v1/collections", json={"name": "v", "dim": 2, "index": "flat"})
    assert client.post("/v1/collections/v/points", json={"vectors": [[1, 2, 3]]}).status_code == 422
    assert (
        client.post("/v1/collections/v/points", json={"vectors": [[1, 2]], "texts": ["a", "b"]}).status_code
        == 422
    )
    client.post("/v1/collections/v/points", json={"vectors": [[1, 2]]})
    assert client.post("/v1/collections/v/search", json={"vector": [1, 2, 3]}).status_code == 422
    assert client.post("/v1/collections/v/hybrid", json={"vector": [1, 2], "text": "x"}).status_code == 422
    assert client.delete("/v1/collections/v/points/99").status_code == 404
    assert client.post("/v1/collections", json={"name": "bad", "dim": 0}).status_code == 422


def test_cli_bench(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "bench.json"
    res = CliRunner().invoke(
        cli_app,
        ["bench", "--n", "600", "--nq", "10", "--dim", "32", "--clusters", "8", "--m", "8",
         "--ef-construction", "32", "--output", str(out)],
    )  # fmt: skip
    assert res.exit_code == 0, res.stdout
    report = json.loads(out.read_text())
    assert report["dataset"]["n"] == 600
    assert all(k["bit_exact"] for k in report["kernels"])
    flat = report["sweep"][0]
    assert flat["index"] == "flat" and flat["recall_at_10"] == 1.0
    hnsw = [p for p in report["sweep"] if p["index"] == "hnsw"]
    assert max(p["recall_at_10"] for p in hnsw) >= 0.95
    assert np.isfinite([p["qps"] for p in report["sweep"]]).all()
    assert "Vecta ANN sweep" in res.stdout
