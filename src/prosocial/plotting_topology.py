"""Plot the structure-vs-shaping cooperation gap across graph topologies.

Reads results/topology.jsonl -> outputs/figures/topology_gap.png.
Gap = mean(interdep coop) - mean(neighbor-shaping coop) vs alpha, one line per
topology. A positive gap on every topology shows the structure>shaping effect
(at first-order-matched beta=alpha) is not specific to one constructed graph.
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


def plot_topology(results="results/topology.jsonl", outdir="outputs/figures"):
    rows = [json.loads(l) for l in open(ROOT / results)]
    outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for i, topo in enumerate(["chain", "ring", "star", "complete"]):
        byk = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["topology"] == topo:
                byk[r["kind"]][r["alpha"]].append(r["coop"])
        alphas = sorted(byk["interdep"])
        gap = [np.mean(byk["interdep"][a]) - np.mean(byk["neighbor_shaping"][a])
               for a in alphas]
        ls = "--" if topo == "complete" else "-"
        ax.plot(alphas, gap, marker="o", ls=ls, color=f"C{i}",
                label=f"{topo}{' (control)' if topo=='complete' else ''}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"coupling $\alpha$ (= shaping $\beta$, first-order matched)")
    ax.set_ylabel("cooperation gap (interdep $-$ shaping)")
    ax.set_title("Structure $>$ first-order-matched shaping across topologies")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f = outdir / "topology_gap.png"
    fig.savefig(f, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-topo] wrote {f.name}")
    return f


if __name__ == "__main__":
    import sys

    plot_topology(sys.argv[1] if len(sys.argv) > 1 else "results/topology.jsonl")
