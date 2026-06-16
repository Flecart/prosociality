"""Plot spatial-dilemma coop signal + welfare vs interdependence alpha.

Reads results/spatial.jsonl -> outputs/figures/spatial_coop_welfare.png.
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


def plot_spatial(results="results/spatial.jsonl", outdir="outputs/figures"):
    rows = [json.loads(l) for l in open(ROOT / results)]
    outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    envs = sorted({r["env"] for r in rows})

    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4.0), squeeze=False)
    for ax, env in zip(axes[0], envs):
        sub = [r for r in rows if r["env"] == env]
        ca = defaultdict(list)
        wa = defaultdict(list)
        for r in sub:
            ca[r["alpha"]].append(r["coop"])
            wa[r["alpha"]].append(r["welfare"])
        xs = sorted(ca)
        coop = [np.mean(ca[a]) for a in xs]
        welf = [np.mean(wa[a]) for a in xs]
        l1, = ax.plot(xs, coop, marker="o", color="C0", label="coop signal")
        ax.set_ylabel("cooperation signal", color="C0")
        ax.set_xlabel(r"interdependence $\alpha$")
        ax.set_title(env)
        ax.set_ylim(-0.05, 1.05)
        ax2 = ax.twinx()
        l2, = ax2.plot(xs, welf, marker="s", color="C3", label="welfare (raw)")
        ax2.set_ylabel("social welfare (raw)", color="C3")
        ax.legend(handles=[l1, l2], fontsize=8, loc="best")
    fig.suptitle("Spatial dilemmas: cooperation & welfare vs interdependence (smoke scale)",
                 y=1.03)
    fig.tight_layout()
    f = outdir / "spatial_coop_welfare.png"
    fig.savefig(f, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-spatial] wrote {f.name}")
    return f


if __name__ == "__main__":
    import sys

    plot_spatial(sys.argv[1] if len(sys.argv) > 1 else "results/spatial.jsonl")
