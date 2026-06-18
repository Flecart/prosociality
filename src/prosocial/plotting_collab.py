"""Figures for collaboration-based endogenous interdependence (the new paper).

Reads the results/collab_*.jsonl files and writes PNGs to paper/figures/:
  collab_bootstrap.png   -- the headline: cooperation & A co-evolve (n=2)
  collab_freerider.png   -- partner-specific care: collab starves the defector
  collab_algos.png       -- the mechanism across IQL/DQN/A2C/PPO
  collab_groupsize.png   -- the feasibility ceiling across group size
  collab_spatial.png     -- the embodied log-hunt (CleanupStag)
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
FIG = ROOT / "paper" / "figures"
C = {"selfish": "#777777", "fixed": "#1f77b4", "collab": "#d62728"}


def _load(name):
    p = ROOT / "results" / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def _smooth(x, w=51):
    """Edge-safe moving average (reflect padding, no boundary roll-off)."""
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="valid")[:len(x)]


def plot_bootstrap():
    rows = _load("collab_matrix_traj.jsonl")
    boot = _load("collab_matrix_bootstrap.jsonl")
    if not rows:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # Panel A: co-evolution of cooperation and learned A01 (collab, n=2)
    ax = axes[0]
    collab = [r for r in rows if r["mode"] == "collab"]
    selfish = [r for r in rows if r["mode"] == "selfish"]
    cc = np.mean([_smooth(r["coop_curve"]) for r in collab], axis=0)
    sc = np.mean([_smooth(r["coop_curve"]) for r in selfish], axis=0)
    ep = np.arange(len(cc))
    ax.plot(ep, sc, color=C["selfish"], label="cooperation (selfish)")
    ax.plot(ep, cc, color=C["collab"], label="cooperation (collab)")
    ax.set_xlabel("episode"); ax.set_ylabel("cooperation rate")
    ax.set_ylim(-0.05, 1.05)
    # learned A01 trajectory (collab), averaged
    A01 = np.mean([[row[2] for row in r["A_traj"]] for r in collab], axis=0)
    eA = [row[0] for row in collab[0]["A_traj"]]
    ax2 = ax.twinx()
    ax2.plot(eA, A01, color="#2ca02c", ls="--", label=r"learned $A_{01}$ (collab)")
    ax2.set_ylabel(r"learned caring $A_{01}$", color="#2ca02c")
    ax2.set_ylim(-0.05, 1.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title("(a) Cooperation self-organizes with caring")

    # Panel B: final cooperation per mode with seed scatter
    ax = axes[1]
    modes = ["selfish", "fixed", "collab"]
    agg = defaultdict(list)
    for r in boot:
        agg[r["mode"]].append(r["coop_rate"])
    for i, m in enumerate(modes):
        vals = agg[m]
        ax.bar(i, np.mean(vals), color=C[m], alpha=0.8,
               yerr=np.std(vals), capsize=4)
        ax.scatter([i] * len(vals), vals, color="k", s=12, zorder=3, alpha=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["selfish\n(A=0)", "fixed\n(α=0.8)", "collab\n(learned A)"])
    ax.set_ylabel("final cooperation rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"(b) 2-player log-hunt ({len(agg['collab'])} seeds)")
    fig.suptitle("Collaboration-based interdependence bootstraps one-shot cooperation",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    f = FIG / "collab_bootstrap.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[collab] wrote {f.name}")


def plot_freerider():
    rows = _load("collab_matrix_freerider.jsonl")
    if not rows:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # Panel A: learner cooperation per mode
    ax = axes[0]
    modes = ["selfish", "fixed", "collab"]
    agg = defaultdict(list)
    for r in rows:
        agg[r["mode"]].append(r["coop_rate"])
    for i, m in enumerate(modes):
        v = agg[m]
        ax.bar(i, np.mean(v), color=C[m], alpha=0.8, yerr=np.std(v), capsize=4)
    ax.set_xticks(range(3)); ax.set_xticklabels(modes)
    ax.set_ylabel("learner cooperation rate"); ax.set_ylim(0, 1.05)
    ax.set_title("(a) cooperation among learners")

    # Panel B: care on partner vs defector (fixed vs collab)
    ax = axes[1]
    A = {m: np.mean([np.array(r["A_final"]) for r in rows if r["mode"] == m], axis=0)
         for m in ("fixed", "collab")}
    x = np.arange(2); w = 0.35
    ax.bar(x - w / 2, [A["fixed"][0, 1], A["fixed"][0, 2]], w,
           color=C["fixed"], label="fixed (α)")
    ax.bar(x + w / 2, [A["collab"][0, 1], A["collab"][0, 2]], w,
           color=C["collab"], label="collab (learned)")
    ax.set_xticks(x); ax.set_xticklabels(["care on\npartner", "care on\ndefector"])
    ax.set_ylabel(r"learned caring weight $A_{ij}$")
    ax.legend(fontsize=9)
    ax.set_title("(b) collab withholds care from the free-rider")
    fig.suptitle("Free-rider (N=3, one frozen defector): partner-specific assortment",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    f = FIG / "collab_freerider.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[collab] wrote {f.name}")


def plot_algos():
    rows = _load("collab_algos.jsonl")
    if not rows:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    learners = ["iql", "dqn", "a2c", "ppo"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # Panel A: bootstrap cooperation per learner x mode
    ax = axes[0]
    boot = [r for r in rows if r["setting"] == "bootstrap"]
    agg = defaultdict(list)
    for r in boot:
        agg[(r["learner"], r["mode"])].append(r["coop_rate"])
    x = np.arange(len(learners)); w = 0.26
    for j, m in enumerate(["selfish", "fixed", "collab"]):
        means = [np.mean(agg[(k, m)]) for k in learners]
        errs = [np.std(agg[(k, m)]) for k in learners]
        ax.bar(x + (j - 1) * w, means, w, yerr=errs, capsize=3,
               color=C[m], label=m, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([k.upper() for k in learners])
    ax.set_ylabel("cooperation rate"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9); ax.set_title("(a) bootstrap across learners (n=2)")

    # Panel B: free-rider care concentration (A_defector ~ 0 for all learners)
    ax = axes[1]
    free = [r for r in rows if r["setting"] == "freerider"]
    agg2 = defaultdict(list)
    for r in free:
        if r["mode"] == "collab":
            A = np.array(r["A_final"])
            agg2[r["learner"]].append((A[0, 1], A[0, 2]))
    part = [np.mean([a[0] for a in agg2[k]]) for k in learners]
    deft = [np.mean([a[1] for a in agg2[k]]) for k in learners]
    ax.bar(x - w / 2, part, w, color="#2ca02c", label="care on partner")
    ax.bar(x + w / 2, deft, w, color="#777777", label="care on defector")
    ax.set_xticks(x); ax.set_xticklabels([k.upper() for k in learners])
    ax.set_ylabel(r"learned caring $A_{ij}$ (collab)")
    ax.legend(fontsize=9)
    ax.set_title("(b) free-rider exclusion is learner-independent")
    fig.suptitle("Collaboration-based interdependence across MARL algorithms",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    f = FIG / "collab_algos.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[collab] wrote {f.name}")


def plot_groupsize():
    rows = _load("collab_matrix_groupsize.jsonl")
    if not rows:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.0))
    ns = sorted({r["n"] for r in rows})
    for m in ["selfish", "fixed", "collab"]:
        means, errs = [], []
        for n in ns:
            v = [r["coop_rate"] for r in rows if r["n"] == n and r["mode"] == m]
            means.append(np.mean(v)); errs.append(np.std(v))
        ax.errorbar(ns, means, yerr=errs, marker="o", color=C[m], label=m, capsize=3)
    ax.plot(ns, [1.0 / (n - 1) for n in ns], "k:", label=r"ceiling $1/(N-1)$")
    ax.set_xlabel("group size N (all-hands stag hunt)")
    ax.set_ylabel("cooperation rate"); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(ns); ax.legend(fontsize=9)
    ax.set_title("Feasibility ceiling limits all-hands cooperation at scale")
    fig.tight_layout()
    f = FIG / "collab_groupsize.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[collab] wrote {f.name}")


def plot_spatial():
    rows = _load("collab_spatial.jsonl")
    if not rows:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, setting, title in [(axes[0], "main", "(a) 4-player log-hunt"),
                               (axes[1], "freerider",
                                "(b) with a frozen free-rider")]:
        agg = defaultdict(list)
        for r in rows:
            if r["setting"] == setting:
                agg[r["mode"]].append(r["joint_lifts"])
        modes = ["selfish", "fixed", "collab"]
        for i, m in enumerate(modes):
            v = agg[m]
            if v:
                ax.bar(i, np.mean(v), color=C[m], alpha=0.8,
                       yerr=np.std(v), capsize=4)
                ax.scatter([i] * len(v), v, color="k", s=12, alpha=0.5, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(modes)
        ax.set_ylabel("joint log-lifts per episode")
        ax.set_title(title)
    fig.suptitle("Embodied log-hunt (CleanupStag, A2C): cooperative lifts vs reward regime",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    f = FIG / "collab_spatial.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[collab] wrote {f.name}")


def plot_revision():
    """Revision figure: graded free-rider, normalization ablation, kappa sweep,
    and the horizon (orthogonality) composition."""
    graded = _load("collab_matrix_graded.jsonl")
    abl = _load("collab_matrix_ablation.jsonl")
    kap = _load("collab_matrix_kappa.jsonl")
    horiz = _load("collab_matrix_horizon.jsonl")
    if not (graded and abl and kap and horiz):
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))

    # (a) graded free-rider: care tracks reciprocation
    ax = axes[0]
    ps = sorted({r["defector_coop_p"] for r in graded})
    part = [np.mean([np.array(r["A_final"])[0, 1] for r in graded
                     if r["defector_coop_p"] == p]) for p in ps]
    free = [np.mean([np.array(r["A_final"])[0, 2] for r in graded
                     if r["defector_coop_p"] == p]) for p in ps]
    frees = [np.std([np.array(r["A_final"])[0, 2] for r in graded
                     if r["defector_coop_p"] == p]) for p in ps]
    ax.plot(ps, part, "o-", color="#2ca02c", label="care on co-learner")
    ax.errorbar(ps, free, yerr=frees, fmt="s-", color="#d62728",
                label="care on graded partner", capsize=3)
    ax.set_xlabel("free-rider cooperation prob. $p$")
    ax.set_ylabel(r"learned caring $A_{ij}$"); ax.set_ylim(-0.05, 1.0)
    ax.legend(fontsize=8); ax.set_title("(a) care tracks reciprocation")

    # (b) normalization ablation: basin-crossing under normalized vs raw
    ax = axes[1]
    modes = ["selfish", "fixed", "collab"]
    x = np.arange(3); w = 0.38
    for j, norm in enumerate([True, False]):
        fr = [np.mean([r["coop_rate"] > 0.5 for r in abl
                       if r["mode"] == m and r["normalize"] == norm]) for m in modes]
        ax.bar(x + (j - 0.5) * w, fr, w,
               label="normalized" if norm else "raw $(I-A)^{-1}$",
               color="#1f77b4" if norm else "#ff7f0e", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_ylabel("fraction of seeds cooperating"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.set_title("(b) normalization ablation")

    # (c) kappa sweep: basin-crossing fraction vs kappa
    ax = axes[2]
    ks = sorted({r["kappa"] for r in kap})
    fr = [np.mean([r["coop_rate"] > 0.5 for r in kap if r["kappa"] == k]) for k in ks]
    xi = np.arange(len(ks))
    ax.plot(xi, fr, "o-", color="#d62728")
    ax.set_xlabel(r"link half-saturation $\kappa$")
    ax.set_ylabel("fraction of seeds cooperating"); ax.set_ylim(0, 1.05)
    ax.set_xticks(xi); ax.set_xticklabels([str(k) for k in ks])
    ax.axvline(1, color="gray", ls=":", lw=1)   # the kappa=0.5 default
    ax.set_title(r"(c) sensitivity to $\kappa$")

    # (d) horizon composition: structural x temporal levers
    ax = axes[3]
    hs = sorted({r["horizon"] for r in horiz})
    # coop_rate sums staggers over the H steps; report per-step cooperation
    for m, col in [("selfish", C["selfish"]), ("collab", C["collab"])]:
        means = [np.mean([r["coop_rate"] / h for r in horiz
                          if r["horizon"] == h and r["mode"] == m]) for h in hs]
        errs = [np.std([r["coop_rate"] / h for r in horiz
                        if r["horizon"] == h and r["mode"] == m]) for h in hs]
        ax.errorbar([str(h) for h in hs], means, yerr=errs, fmt="o-", color=col,
                    label=m, capsize=3)
    ax.set_xlabel("horizon H (repetition)")
    ax.set_ylabel("cooperation rate"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8); ax.set_title("(d) structural + temporal compose")
    fig.suptitle("Robustness & ablations: graded free-rider, normalization, $\\kappa$, and Folk-Theorem orthogonality",
                 y=1.04, fontsize=12)
    fig.tight_layout()
    f = FIG / "collab_revision.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[collab] wrote {f.name}")


def main():
    plot_bootstrap()
    plot_freerider()
    plot_algos()
    plot_groupsize()
    plot_spatial()
    plot_revision()


if __name__ == "__main__":
    main()
