"""Figure: dissociating interdependence (alpha) from far-sightedness (gamma).

Reads results/commons_dissociation.jsonl (2x2 mechanism cells x externality knob
xi) and shows the double dissociation:

  (a) cooperation vs xi -- the two single-mechanism curves CROSS: interdependence
      cooperates only when the externality is contemporaneous (xi->0), far-
      sightedness only when it is intertemporal (xi->1).
  (b) the 2x2 at the extremes -- each mechanism cooperates in exactly one regime,
      ruling out reducing either to the other.

Run:
    PYTHONPATH=src uv run --with numpy --with matplotlib \
        python -m prosocial.plotting_commons
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "commons_dissociation.jsonl"
FIGDIR = ROOT / "outputs" / "figures"

STYLE = {                       # (label, colour, linestyle)
    "selfish_myopic":      ("selfish + myopic  (neither)", "#9aa0a6", ":"),
    "interdep_myopic":     ("interdependent + myopic  ($\\alpha$ only)", "#1f77b4", "-"),
    "selfish_farsighted":  ("selfish + far-sighted  ($\\gamma$ only)", "#d1495b", "-"),
    "interdep_farsighted": ("interdependent + far-sighted  (both)", "#2a9d8f", "--"),
}


def load(path=RESULTS):
    rows = [json.loads(l) for l in open(path)]
    data = {}
    for r in rows:
        data.setdefault(r["cell"], {})[r["xi"]] = (r["coop"], r["coop_std"])
    return data


def make_figure(out=None):
    data = load()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5),
                           gridspec_kw={"width_ratios": [1.6, 1]})

    # -- (a) cooperation vs xi: the crossing -------------------------------- #
    for cell, (label, col, ls) in STYLE.items():
        if cell not in data:
            continue
        xis = sorted(data[cell])
        m = np.array([data[cell][x][0] for x in xis])
        s = np.array([data[cell][x][1] for x in xis])
        ax[0].plot(xis, m, ls, color=col, lw=2.4, marker="o", ms=5, label=label)
        ax[0].fill_between(xis, m - s, m + s, color=col, alpha=0.13)
    ax[0].set_xlabel("externality routing  $\\xi$\n"
                     "hurts others NOW $\\;\\longrightarrow\\;$ hurts MY future")
    ax[0].set_ylabel("cooperation  (restraint = abstain fraction)")
    ax[0].set_ylim(-0.03, 1.05)
    ax[0].set_title("(a) Two mechanisms, two regimes: the curves cross")
    ax[0].legend(frameon=False, fontsize=9, loc="center left",
                 bbox_to_anchor=(0.0, 0.55))
    ax[0].annotate("interdependence\ncures THIS end",
                   xy=(0.0, 1.0), xytext=(0.13, 0.80), fontsize=8.5,
                   color="#1f77b4", ha="left")
    ax[0].annotate("far-sightedness\ncures THIS end",
                   xy=(1.0, 0.61), xytext=(0.60, 0.30), fontsize=8.5,
                   color="#d1495b", ha="left")

    # -- (b) the 2x2 at the extremes ---------------------------------------- #
    xis_all = sorted(next(iter(data.values())))
    xlo, xhi = xis_all[0], xis_all[-1]
    cells = ["selfish_myopic", "interdep_myopic", "selfish_farsighted",
             "interdep_farsighted"]
    short = ["neither", "$\\alpha$ only", "$\\gamma$ only", "both"]
    lo = [data[c][xlo][0] for c in cells]
    hi = [data[c][xhi][0] for c in cells]
    x = np.arange(4); w = 0.38
    ax[1].bar(x - w / 2, lo, w, color="#5b8def",
              label=f"$\\xi$={xlo:g}  (contemporaneous)")
    ax[1].bar(x + w / 2, hi, w, color="#e9a23b",
              label=f"$\\xi$={xhi:g}  (intertemporal)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(short)
    ax[1].set_ylabel("cooperation")
    ax[1].set_ylim(0, 1.08)
    ax[1].set_title("(b) Each mechanism works in exactly one regime")
    ax[1].legend(frameon=False, fontsize=8.5, loc="upper center")

    fig.suptitle("Interdependence ($\\alpha$) vs far-sightedness ($\\gamma$): a clean "
                 "double dissociation in a tunable-externality commons\n"
                 "(no partner observation $\\Rightarrow$ no reciprocity; cooperation "
                 "comes only from caring-about-others or caring-about-future-self)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = out or (FIGDIR / "interdependence_vs_farsightedness.png")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")
    return out


if __name__ == "__main__":
    make_figure()
