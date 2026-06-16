"""Plot structure-vs-shaping on a non-complete (chain) graph.

Reads results/graph_structure.jsonl -> outputs/figures/graph_structure.png.
Chain interdependence vs first-order-matched neighbor shaping on a 3-player
chain Public Goods Game; the gap is the empirical signature of indirect coupling.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"chain_interdep": r"chain interdependence $(I-A)^{-1}\pi$",
          "neighbor_shaping": r"neighbor shaping $\pi_i+\alpha\!\sum_{N(i)}\pi_j$"}


def plot_graph(results="results/graph_structure.jsonl", outdir="outputs/figures"):
    rows = [json.loads(l) for l in open(ROOT / results)]
    outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fig, (axc, axw) = plt.subplots(1, 2, figsize=(10, 4.0))
    for i, k in enumerate(["chain_interdep", "neighbor_shaping"]):
        byA = defaultdict(list)
        for r in rows:
            if r["kind"] == k:
                byA[r["alpha"]].append(r)
        xs = sorted(byA)
        c = np.array([np.mean([x["coop"] for x in byA[a]]) for a in xs])
        cs = np.array([np.std([x["coop"] for x in byA[a]]) for a in xs])
        w = np.array([np.mean([x["welfare"] for x in byA[a]]) for a in xs])
        axc.plot(xs, c, marker="o", color=f"C{i}", label=LABELS[k])
        axc.fill_between(xs, c - cs, c + cs, color=f"C{i}", alpha=0.12)
        axw.plot(xs, w, marker="s", color=f"C{i}", label=LABELS[k])
    for ax in (axc, axw):
        ax.set_xlabel(r"coupling $\alpha$ (= shaping $\beta$, first-order matched)")
        ax.legend(fontsize=8)
    axc.set_ylabel("cooperation rate"); axc.set_ylim(-0.05, 1.05)
    axc.set_title("Cooperation (3-player chain PGG)")
    axw.set_ylabel("social welfare (raw)")
    axw.set_title("Welfare")
    fig.suptitle("Structure $\\neq$ shaping on a non-complete graph: indirect "
                 "coupling helps", y=1.02)
    fig.tight_layout()
    f = outdir / "graph_structure.png"
    fig.savefig(f, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-graph] wrote {f.name}")
    return f


if __name__ == "__main__":
    import sys

    plot_graph(sys.argv[1] if len(sys.argv) > 1 else "results/graph_structure.jsonl")
