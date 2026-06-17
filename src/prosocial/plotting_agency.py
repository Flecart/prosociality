"""Error-management figure for the agency-attribution experiment (Coin Game).

Reads results/agency_coin.jsonl and tells the error-management story:

  (a) Measured payoff structure. A forest (F) and a rock (K) are both inert and
      look identical to an agency detector, but caring is RIGHT for the forest
      (respecting it pays) and WRONG for the rock (respecting wastes free coins).
  (b) The decision. Given a noisy cue that only weakly separates forest from
      rock, the focal confers care iff cue > theta. The fitness-optimal theta
      depends on how common forests are: when forests are common the optimum
      shifts toward OVER-attribution ("animism").
  (c) The phase diagram. Over (P(forest), cost-asymmetry) the optimal policy is
      either skeptic or animist; the measured Coin-Game operating point is marked.
  (d) The learner. An A2C agent (trained on the GPU) that attributes more agency
      (higher Bergstrom alpha) sustains the renewable forest and eats more.

Run:
    PYTHONPATH=src uv run --with numpy --with matplotlib \
        python -m prosocial.plotting_agency
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "agency_coin.jsonl"
FIGDIR = ROOT / "outputs" / "figures"


def _phi(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def load(path=RESULTS):
    rows = [json.loads(l) for l in open(path)]
    mat = {(r["ptype"], r["strategy"]): (r["focal_raw"], r["focal_raw_std"])
           for r in rows if r.get("kind") == "matrix"}
    learn = sorted([r for r in rows if r.get("kind") == "learn"],
                   key=lambda r: r["alpha"])
    return mat, learn


def expected_fitness(mat, thetas, p_forest, mu_F=1.0, mu_K=0.0, sigma=0.8):
    """E[focal RAW fitness] vs detection threshold theta, given a fraction
    p_forest of forests among the inert entities. Confer (respect) iff cue>theta;
    forests emit cue~N(mu_F,sigma), rocks cue~N(mu_K,sigma) (overlapping: weak
    signal). Respect pays Delta_F for a forest, Delta_K(<0) for a rock."""
    dF = mat[("F", "respect")][0] - mat[("F", "exploit")][0]
    dK = mat[("K", "respect")][0] - mat[("K", "exploit")][0]
    base = (p_forest * mat[("F", "exploit")][0]
            + (1 - p_forest) * mat[("K", "exploit")][0])
    p_resp_F = 1.0 - _phi((thetas - mu_F) / sigma)
    p_resp_K = 1.0 - _phi((thetas - mu_K) / sigma)
    E = base + p_forest * p_resp_F * dF + (1 - p_forest) * p_resp_K * dK
    # liberality = P(confer | a typical inert entity) at this theta
    lib = 0.5 * (p_resp_F + p_resp_K)
    return E, lib


def make_figure(out=None):
    mat, learn = load()
    dF = mat[("F", "respect")][0] - mat[("F", "exploit")][0]
    dK = mat[("K", "respect")][0] - mat[("K", "exploit")][0]
    print(f"[plot] Delta_F={dF:+.3f} (miss-forest cost)  "
          f"Delta_K={dK:+.3f} (waste-on-rock cost={-dK:+.3f})")

    thetas = np.linspace(-2.5, 3.0, 500)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # -- (a) measured payoff matrix ------------------------------------------- #
    labels = ["forest F\n(renewable)", "rock K\n(barren)"]
    resp = [mat[("F", "respect")][0], mat[("K", "respect")][0]]
    resp_e = [mat[("F", "respect")][1], mat[("K", "respect")][1]]
    expl = [mat[("F", "exploit")][0], mat[("K", "exploit")][0]]
    expl_e = [mat[("F", "exploit")][1], mat[("K", "exploit")][1]]
    x = np.arange(2); w = 0.38
    ax[0].bar(x - w / 2, resp, w, yerr=resp_e, capsize=4, label="respect (confer agency)",
              color="#2a9d8f")
    ax[0].bar(x + w / 2, expl, w, yerr=expl_e, capsize=4, label="exploit (no agency)",
              color="#e76f51")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("focal RAW fitness")
    ax[0].set_title("(a) Same cue, opposite payoff:\nrespect helps F, wastes on K")
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    ax[0].annotate(f"$\\Delta_F$={dF:+.1f}", (0, max(resp[0], expl[0])),
                   xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax[0].annotate(f"$\\Delta_K$={dK:+.1f}", (1, max(resp[1], expl[1])),
                   xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)

    # -- (b) optimal bias vs how common forests are --------------------------- #
    for label, pf, col in [("forests rare (10%)", 0.1, "#264653"),
                           ("balanced (50%)", 0.5, "#2a9d8f"),
                           ("forests common (90%)", 0.9, "#e9c46a")]:
        E, lib = expected_fitness(mat, thetas, pf)
        ax[1].plot(lib, E, color=col, lw=2, label=label)
        j = int(np.argmax(E))
        ax[1].plot(lib[j], E[j], "o", color=col, ms=8)
    ax[1].set_xlabel("agency liberality  P(confer | inert entity)\n"
                     "skeptic $\\longrightarrow$ animist")
    ax[1].set_ylabel("expected focal RAW fitness")
    ax[1].set_title("(b) Optimal bias (●) moves toward\nover-attribution as forests"
                    " get common")
    ax[1].legend(frameon=False, fontsize=8.5, loc="best")

    # -- (c) phase diagram: optimal liberality over (P(forest), cost ratio) --- #
    pf_grid = np.linspace(0.02, 0.98, 60)
    # cost ratio r = |Delta_K| / Delta_F ; scale Delta_K to sweep asymmetry
    ratio_grid = np.linspace(0.2, 6.0, 60)
    opt_lib = np.zeros((len(ratio_grid), len(pf_grid)))
    matc = dict(mat)
    for ri, r in enumerate(ratio_grid):
        # rebuild a matrix with |Delta_K| = r * Delta_F (keep exploit baselines)
        matc[("K", "respect")] = (mat[("K", "exploit")][0] - r * dF, 0)
        for pi, pf in enumerate(pf_grid):
            E, lib = expected_fitness(matc, thetas, pf)
            opt_lib[ri, pi] = lib[int(np.argmax(E))]
    im = ax[2].imshow(opt_lib, origin="lower", aspect="auto", cmap="RdYlGn",
                      extent=[pf_grid[0], pf_grid[-1], ratio_grid[0], ratio_grid[-1]],
                      vmin=0, vmax=1)
    ax[2].set_xlabel("fraction of inert entities that are forests")
    ax[2].set_ylabel("cost asymmetry  |$\\Delta_K$| / $\\Delta_F$")
    # mark the measured Coin-Game operating point (at its true cost ratio)
    ax[2].plot(0.5, abs(dK) / dF, "k*", ms=16, label="measured Coin Game")
    ax[2].axhline(abs(dK) / dF, color="k", lw=0.6, ls=":")
    ax[2].legend(frameon=False, fontsize=8.5, loc="upper right")
    ax[2].set_title("(c) Phase diagram: optimal agency\nliberality (green=animist)")
    cb = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    cb.set_label("optimal liberality", fontsize=8)

    fig.suptitle("Adaptive agency mis-attribution in the Coin Game — error-management "
                 "theory  (forest gives back; missing it is the costly error)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out or (FIGDIR / "agency_coin_error_management.png")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"[plot] wrote {out}")
    return out


if __name__ == "__main__":
    make_figure()
