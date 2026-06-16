"""Plots for endogenous interdependence learning.

Reads results/endogenous_*.jsonl -> outputs/figures/endogenous.png. Three panels:
  (a) self-play alpha emergence: naive (per-direction -> asymmetric) vs mutual
      (shared -> symmetric, high);
  (b) welfare over outer steps (both rise; mutual reaches the cooperative optimum);
  (c) directed caring: mean alpha toward a co-learner vs toward a fixed defector.
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


def _load(name):
    p = ROOT / f"results/endogenous_{name}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p)]


def _by_t(rows, fn):
    """mean of fn(row) grouped by outer step t (averaged over seeds)."""
    d = defaultdict(list)
    for r in rows:
        d[r["t"]].append(fn(r))
    ts = sorted(d)
    return np.array(ts), np.array([np.mean(d[t], axis=0) for t in ts])


def plot_endogenous(outdir="outputs/figures"):
    outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    naive = _load("naive2")
    recip = _load("reciprocal2")
    mutual = _load("mutual2")
    mixed = _load("mixed3")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) alpha trajectories: three regimes (all at delta=0.5)
    ax = axes[0]
    if naive:
        ts, a01 = _by_t(naive, lambda r: r["A"][0][1])
        _, a10 = _by_t(naive, lambda r: r["A"][1][0])
        ax.plot(ts, (a01 + a10) / 2, color="C7", label="self-interest (collapse)")
    if recip:
        ts, a01 = _by_t(recip, lambda r: max(r["A"][0][1], r["A"][1][0]))
        _, a10 = _by_t(recip, lambda r: min(r["A"][0][1], r["A"][1][0]))
        ax.plot(ts, a01, color="C3", label=r"reciprocal: carer")
        ax.plot(ts, a10, color="C3", ls="--", label=r"reciprocal: free-rider")
    if mutual:
        ts, a01 = _by_t(mutual, lambda r: r["A"][0][1])
        ax.plot(ts, a01, color="C0", lw=2, label=r"joint-payoff commitment")
    ax.axhline(2 / 3, ls=":", color="k", lw=1)
    ax.text(ax.get_xlim()[1] * 0.5, 0.68, r"$\alpha^*=2/3$", fontsize=8)
    ax.set_xlabel("outer step (slow timescale)")
    ax.set_ylabel(r"learned caring $\alpha$")
    ax.set_title("(a) Three regimes of learned caring")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.0)

    # (b) welfare
    ax = axes[1]
    for rows, c, lab in [(naive, "C7", "self-interest"),
                         (recip, "C3", "reciprocal (asymmetric)"),
                         (mutual, "C0", "joint-payoff commitment")]:
        if rows:
            ts, w = _by_t(rows, lambda r: r["welfare"])
            ax.plot(ts, w, color=c, label=lab)
    ax.axhline(6.0, ls=":", color="k", lw=1)
    ax.text(1, 6.05, "cooperative optimum", fontsize=8)
    ax.set_xlabel("outer step")
    ax.set_ylabel("social welfare (raw)")
    ax.set_title("(b) Welfare: only joint commitment reaches optimum")
    ax.legend(fontsize=8)

    # (c) directed caring (mixed3): co-learner vs defector
    ax = axes[2]
    if mixed:
        # learners are 0,1; defector is 2. Average final-window alpha.
        last = [r for r in mixed if r["t"] >= max(rr["t"] for rr in mixed) - 7]
        to_learner, to_def = [], []
        for r in last:
            A = r["A"]
            to_learner += [A[0][1], A[1][0]]
            to_def += [A[0][2], A[1][2]]
        means = [np.mean(to_learner), np.mean(to_def)]
        sds = [np.std(to_learner), np.std(to_def)]
        ax.bar(["toward\nco-learner", "toward\ndefector"], means, yerr=sds,
               color=["C0", "C3"], capsize=4)
        ax.set_ylabel(r"learned caring $\alpha$")
        ax.set_title("(c) Directed: care for reciprocators, not exploiters")
    fig.suptitle("Endogenous interdependence learning: agents learn whom to care about",
                 y=1.03)
    fig.tight_layout()
    f = outdir / "endogenous.png"
    fig.savefig(f, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-endo] wrote {f.name}")
    return f


if __name__ == "__main__":
    plot_endogenous()
