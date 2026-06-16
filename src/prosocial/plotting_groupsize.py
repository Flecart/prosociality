"""Plot the group-size feasibility law: cooperation onset vs N against 1/(N-1).

Reads results/group_size.jsonl -> outputs/figures/group_size_law.png.
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


def plot_groupsize(results="results/group_size.jsonl", outdir="outputs/figures"):
    rows = [json.loads(l) for l in open(ROOT / results)]
    outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    Ns = sorted({r["n_agents"] for r in rows})
    onset, ceiling = [], []
    for n in Ns:
        byA = defaultdict(list)
        for r in rows:
            if r["n_agents"] == n:
                byA[r["alpha"]].append(r["coop"])
        on = next((a for a in sorted(byA) if np.mean(byA[a]) > 0.5), np.nan)
        onset.append(on)
        ceiling.append(1.0 / (n - 1))

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    xs = np.array(Ns)
    ax.plot(xs, ceiling, "k--", marker="s", label=r"ceiling $\alpha_{\max}=1/(N-1)$")
    ax.plot(xs, onset, "C0-", marker="o", label="empirical coop onset")
    ax.fill_between(xs, onset, ceiling, color="C0", alpha=0.1)
    ax.set_xlabel("group size $N$")
    ax.set_ylabel(r"interdependence $\alpha$")
    ax.set_title("Group-size feasibility law (one-shot Public Goods)")
    ax.set_xticks(xs)
    ax.legend()
    fig.tight_layout()
    f = outdir / "group_size_law.png"
    fig.savefig(f, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-group] N={Ns} onset={[round(o,3) for o in onset]} "
          f"ceiling={[round(c,3) for c in ceiling]} -> {f.name}")
    return f


if __name__ == "__main__":
    import sys

    plot_groupsize(sys.argv[1] if len(sys.argv) > 1 else "results/group_size.jsonl")
