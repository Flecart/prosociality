"""Plot learner-robustness of the one-shot IPD transition.

Reads results/learner_robust.jsonl -> outputs/figures/learner_robust.png.
Shows cooperation vs alpha for three learners; a sharp transition persists under
all three (its location shifts modestly), so the transition is not an artifact of
one exploration scheme.
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
LABELS = {"egreedy_anneal": r"IQL $\epsilon$-greedy (0.5$\to$0.02)",
          "egreedy_low": r"IQL $\epsilon$-greedy (0.1$\to$0)",
          "boltzmann": "IQL Boltzmann (T=0.3)",
          "a2c": "A2C (on-policy, neural)"}
ORDER = ["egreedy_anneal", "egreedy_low", "boltzmann", "a2c"]


def plot_robust(results="results/learner_robust.jsonl", outdir="outputs/figures"):
    rows = [json.loads(l) for l in open(ROOT / results)]
    # fold in the on-policy A2C learner if present
    a2c_path = ROOT / "results/a2c_ipd.jsonl"
    if a2c_path.exists():
        rows += [json.loads(l) for l in open(a2c_path)]
    outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for i, L in enumerate(ORDER):
        byA = defaultdict(list)
        for r in rows:
            if r["learner"] == L:
                byA[r["alpha"]].append(r["coop"])
        xs = sorted(byA)
        m = np.array([np.mean(byA[a]) for a in xs])
        s = np.array([np.std(byA[a]) for a in xs])
        ax.plot(xs, m, marker="o", color=f"C{i}", label=LABELS[L])
        ax.fill_between(xs, m - s, m + s, color=f"C{i}", alpha=0.12)
    ax.axvline(2 / 3, ls="--", color="k", lw=1)
    ax.text(2 / 3, 1.02, r"$\alpha^*=2/3$", ha="center", fontsize=9)
    ax.set_xlabel(r"interdependence $\alpha$")
    ax.set_ylabel("cooperation rate")
    ax.set_title("One-shot IPD transition is robust across learners")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    f = outdir / "learner_robust.png"
    fig.savefig(f, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-robust] wrote {f.name}")
    return f


if __name__ == "__main__":
    import sys

    plot_robust(sys.argv[1] if len(sys.argv) > 1 else "results/learner_robust.jsonl")
