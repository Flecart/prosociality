"""Emit LaTeX result macros for Table 1 from the full phase-transition sweep.

Prints mean +/- std (over seeds) of one-shot cooperation at alpha=0 and at the
max feasible alpha, for each game. Paste the printed block into content.tex.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _ci(vals):
    vals = np.asarray(vals, float)
    m = vals.mean()
    s = vals.std()
    return m, s


def main(results="results/phase_transition.jsonl"):
    rows = [json.loads(l) for l in open(ROOT / results)]
    macros = {}
    onset = {}
    for game, zkey, mkey in [("IPD", "PHIPDz", "PHIPDm"),
                             ("StagHunt", "PHSHz", "PHSHm"),
                             ("PGG", "PHPGGz", "PHPGGm")]:
        sub = [r for r in rows if r["game"] == game and r["horizon"] == 1
               and r["family"] == "interdep"]
        byA = defaultdict(list)
        for r in sub:
            byA[r["param"]].append(r["coop"])
        alphas = sorted(byA)
        a0 = alphas[0]
        amax = alphas[-1]
        m0, s0 = _ci(byA[a0])
        mm, sm = _ci(byA[amax])
        macros[zkey] = f"{m0:.2f}\\,\\small$\\pm${s0:.2f}"
        macros[mkey] = f"{mm:.2f}\\,\\small$\\pm${sm:.2f}"
        # empirical onset (first alpha with mean>0.5)
        on = next((a for a in alphas if np.mean(byA[a]) > 0.5), None)
        onset[game] = (on, amax)

    print("% --- auto-generated from full sweep; paste into content.tex ---")
    for k, v in macros.items():
        print(f"\\renewcommand{{\\{k}}}{{{v}}}")
    print("% onsets:", {g: onset[g] for g in onset})


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "results/phase_transition.jsonl")
