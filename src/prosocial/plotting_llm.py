"""Plot LLM agent cooperation vs in-context interdependence alpha, per model.

Reads results/llm_games.jsonl -> outputs/figures/llm_coop_vs_alpha.png.
Models recorded as status='unavailable' are listed in the caption, not plotted.
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


def plot_llm(results="results/llm_games.jsonl", outdir="outputs/figures"):
    rows = [json.loads(l) for l in open(ROOT / results)]
    outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    unavailable = sorted({r["model"] for r in rows if r.get("status") == "unavailable"})
    games = sorted({r["game"] for r in ok})
    models = sorted({r["model"] for r in ok})

    if not ok:
        print("[plot-llm] no usable rows; models unavailable:", unavailable)
        return None

    fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 4.2), squeeze=False)
    for ax, game in zip(axes[0], games):
        for mi, model in enumerate(models):
            sub = [r for r in ok if r["game"] == game and r["model"] == model]
            agg = defaultdict(list)
            for r in sub:
                if not np.isnan(r["coop"]):
                    agg[r["alpha"]].append(r["coop"])
            xs = sorted(agg)
            ys = [np.mean(agg[a]) for a in xs]
            ax.plot(xs, ys, marker="o", label=model.split("/")[-1], color=f"C{mi}")
        ax.set_title(f"{game}")
        ax.set_xlabel(r"in-context interdependence $\alpha$")
        ax.set_ylabel("cooperation rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    cap = "LLM cooperation vs prompted interdependence."
    if unavailable:
        cap += "  Unavailable: " + ", ".join(m.split("/")[-1] for m in unavailable)
    fig.suptitle(cap, y=1.02)
    fig.tight_layout()
    f = outdir / "llm_coop_vs_alpha.png"
    fig.savefig(f, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-llm] wrote {f.name}; unavailable={unavailable}")
    return f


if __name__ == "__main__":
    import sys

    plot_llm(sys.argv[1] if len(sys.argv) > 1 else "results/llm_games.jsonl")
