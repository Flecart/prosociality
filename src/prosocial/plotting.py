"""Plots for the phase-transition experiment. Reads results/*.jsonl -> figures/.

Matplotlib only (no seaborn), Agg backend so it runs headless on a compute node.
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
COOP_THRESHOLD = 2.0 / 3.0


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _agg(rows, keyfn, valfn):
    """mean/std of valfn grouped by keyfn -> {key: (xs, means, stds)}."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        gk, x = keyfn(r)
        buckets[gk][x].append(valfn(r))
    out = {}
    for gk, xv in buckets.items():
        xs = sorted(xv)
        means = np.array([np.mean(xv[x]) for x in xs])
        stds = np.array([np.std(xv[x]) for x in xs])
        out[gk] = (np.array(xs), means, stds)
    return out


def plot_phase_transition(results="results/phase_transition.jsonl",
                          outdir="outputs/figures"):
    rows = load(ROOT / results)
    outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    games = sorted({r["game"] for r in rows})

    # --- Fig 1: coop rate vs alpha, one panel per game, line per horizon ---
    fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 4.2), squeeze=False)
    for ax, game in zip(axes[0], games):
        sub = [r for r in rows if r["game"] == game and r["family"] == "interdep"]
        series = _agg(sub, lambda r: (r["horizon"], r["param"]), lambda r: r["coop"])
        for H in sorted(series):
            xs, m, s = series[H]
            ax.plot(xs, m, marker="o", label=f"H={H}")
            ax.fill_between(xs, m - s, m + s, alpha=0.15)
        n = sub[0]["n_agents"]
        amax = 1.0 / (n - 1)
        if game == "IPD":
            ax.axvline(COOP_THRESHOLD, ls="--", color="k", lw=1)
            ax.text(COOP_THRESHOLD, 1.02, r"$\alpha^*=2/3$", ha="center", fontsize=9)
        ax.axvline(amax, ls=":", color="red", lw=1)
        ax.set_title(f"{game} (N={n}, feasible $\\alpha<{amax:.2f}$)")
        ax.set_xlabel(r"interdependence $\alpha$")
        ax.set_ylabel("cooperation rate")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)
    fig.suptitle("Cooperation phase transition under structural interdependence", y=1.02)
    fig.tight_layout()
    f1 = outdir / "phase_transition_coop.png"
    fig.savefig(f1, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: interdependence vs reward-shaping (one-shot) ---
    fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 4.2), squeeze=False)
    for ax, game in zip(axes[0], games):
        for fam, color in [("interdep", "C0"), ("shaping", "C3")]:
            sub = [r for r in rows if r["game"] == game and r["family"] == fam
                   and r["horizon"] == 1]
            if not sub:
                continue
            series = _agg(sub, lambda r: (fam, r["param"]), lambda r: r["coop"])
            xs, m, s = series[fam]
            label = r"interdependence $(I-A)^{-1}\pi$" if fam == "interdep" else r"shaping $\pi_i+\beta\sum\pi_j$"
            ax.plot(xs, m, marker="o", color=color, label=label)
            ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15)
        ax.set_title(f"{game}: one-shot (H=1)")
        ax.set_xlabel(r"coupling parameter ($\alpha$ or $\beta$)")
        ax.set_ylabel("cooperation rate")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)
    fig.suptitle("Structural interdependence vs. flat reward shaping (one-shot)", y=1.02)
    fig.tight_layout()
    f2 = outdir / "interdep_vs_shaping.png"
    fig.savefig(f2, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: welfare vs alpha (IPD), one-shot vs repeated mechanisms ---
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sub = [r for r in rows if r["game"] == "IPD" and r["family"] == "interdep"]
    series = _agg(sub, lambda r: (r["horizon"], r["param"]), lambda r: r["welfare"])
    for H in sorted(series):
        xs, m, s = series[H]
        ax.plot(xs, m, marker="s", label=f"H={H}")
    ax.set_title("IPD: social welfare (raw payoff) vs interdependence")
    ax.set_xlabel(r"interdependence $\alpha$")
    ax.set_ylabel("social welfare / round")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f3 = outdir / "ipd_welfare.png"
    fig.savefig(f3, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {f1.name}, {f2.name}, {f3.name} -> {outdir}")
    return [f1, f2, f3]


if __name__ == "__main__":
    import sys

    res = sys.argv[1] if len(sys.argv) > 1 else "results/phase_transition.jsonl"
    plot_phase_transition(results=res)
