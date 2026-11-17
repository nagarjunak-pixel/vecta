"""``vecta`` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from vecta.api import create_app
from vecta.bench import run_benchmark

app = typer.Typer(
    name="vecta",
    help="Vecta: HNSW / IVF-PQ / binary vector search engine.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command("bench")
def bench_cmd(
    n: int = typer.Option(20_000, min=100, help="Base vectors"),
    nq: int = typer.Option(200, min=1, help="Queries"),
    dim: int = typer.Option(96, min=8, help="Dimensionality (multiple of 16)"),
    clusters: int = typer.Option(64, min=1),
    m: int = typer.Option(16, min=2, help="HNSW M"),
    ef_construction: int = typer.Option(100, min=4),
    output: Path | None = typer.Option(None, help="Write the JSON report here"),
) -> None:
    """Recall@10 / QPS / bytes-per-vector sweep plus native-vs-NumPy kernel timings."""
    report = run_benchmark(n=n, nq=nq, dim=dim, clusters=clusters, hnsw_m=m, ef_construction=ef_construction)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ds = report["dataset"]
    table = Table(title=f"Vecta ANN sweep: {ds['name']} ({ds['nq']} queries, k=10)")
    for col in ("index", "params", "recall@10", "QPS", "bytes/vec", "exact dists/q"):
        table.add_column(col, justify="right" if col not in ("index", "params") else "left")
    for p in report["sweep"]:
        table.add_row(
            p["index"],
            p["params"],
            f"{p['recall_at_10']:.3f}",
            f"{p['qps']:.0f}",
            str(p["bytes_per_vector"]),
            f"{p['distance_evals_per_query']:.0f}",
        )
    console.print(table)
    kt = Table(title=f"Native kernels ({report['native_backend']}) vs NumPy twins")
    for col in ("kernel", "native ms", "numpy ms", "speedup", "bit-exact"):
        kt.add_column(col)
    for kp in report["kernels"]:
        kt.add_row(
            kp["kernel"],
            f"{kp['native_ms']:.2f}",
            f"{kp['numpy_ms']:.2f}",
            f"{kp['speedup']:.1f}x",
            str(kp["bit_exact"]),
        )
    console.print(kt)
    g = report["hnsw_graph"]
    console.print(
        f"HNSW graph: {g['nodes']:.0f} nodes, max level {g['max_level']:.0f}, mean L0 degree "
        f"{g['mean_degree_l0']:.1f}, reachable {g['reachable_fraction']:.3f}; filtered (even ids) "
        f"recall@10 = {report['hnsw_filtered_even_ids_recall_at_10_ef64']:.3f}"
    )


@app.command("serve")
def serve_cmd(host: str = "127.0.0.1", port: int = 8095) -> None:  # pragma: no cover - blocking
    """Serve the collections API."""
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
