"""Endogenous interdependence learning: agents LEARN whom to care about.

The relational matrix A is no longer fixed/exogenous: each agent i carries a
learnable row alpha_{ij} >= 0 (how much it internalizes agent j's welfare).
Two-timescale learning (plan Experiment 3):

  * FAST (inner loop): given the current A, agents learn policies on the
    *transformed* reward U = (I-A)^{-1} pi  (standard tabular IQL).
  * SLOW (outer loop): each agent updates alpha_{ij} by gradient ASCENT on its
    own *raw* payoff pi_i, estimated by finite differences --
        alpha_{ij} <- clip(alpha_{ij} + eta * d pi_i / d alpha_{ij}, 0, cap).
    i.e. an agent comes to care about j only insofar as caring raises its OWN
    material outcome, through the equilibrium it induces.

What actually happens (see paper Sec. "Endogenous Interdependence"):
  - DECENTRALIZED self-interest (naive2, reciprocal2): alpha does rise from 0,
    but to an ASYMMETRIC profile -- one agent becomes the carer, the other free-
    rides (the Samaritan's Dilemma emerging endogenously). Assortment-in-the-probe
    does not fix this; the per-agent update is still selfish.
  - SYMMETRIC mutual care requires a joint-welfare commitment (mutual2): a SHARED
    alpha updated on the JOINT payoff pi_i+pi_j. This is a stronger, relational
    assumption that IMPOSES symmetry by construction -- it shows what structure is
    needed for symmetric internalized care, not that it emerges from self-interest.
  - vs a fixed defector (mixed3): caring about a non-reciprocator lowers your own
    raw payoff, so alpha toward it is driven to ~0. Agents thus learn to give
    prosociality SELECTIVELY -- to reciprocators, not exploiters.

Writes results/endogenous_<setting>.jsonl (alpha trajectory + coop/welfare).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..agents import TabularQLearner
from ..envs import RepeatedGame, make_game
from ..interdependence import effective_utilities, spectral_radius

ROOT = Path(__file__).resolve().parents[3]


def _safe_A(A, cap=0.95):
    """Keep rho(A) < 1 so (I-A)^{-1} is well-posed."""
    A = np.clip(A, 0.0, cap)
    np.fill_diagonal(A, 0.0)
    rho = spectral_radius(A)
    if rho >= 0.97:
        A = A * (0.97 / rho)
    return A


def _train_inner(game_name, A, horizon, episodes, seed, Q_init=None,
                 lr=0.1, eps=0.1, game_kwargs=None, fixed_policies=None):
    """Train tabular IQL on transformed reward U=(I-A)^{-1}pi; return Q tables.

    fixed_policies: dict {agent_idx: 'defect'|'cooperate'} for non-learning agents.
    """
    rng = np.random.default_rng(seed)
    base = make_game(game_name, **(game_kwargs or {}))
    env = RepeatedGame(base, horizon=horizon)
    n = env.n_agents
    agents = []
    for i in range(n):
        ag = TabularQLearner(env.n_states, env.n_actions, lr=lr, rng=rng,
                             eps_start=eps, eps_end=eps)
        if Q_init is not None:
            ag.Q = Q_init[i].copy()
        agents.append(ag)
    fixed_policies = fixed_policies or {}
    coop_idx = base.coop_action
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            acts = []
            for i in range(n):
                if i in fixed_policies:
                    acts.append(coop_idx if fixed_policies[i] == "cooperate"
                                else (1 - coop_idx))
                else:
                    acts.append(agents[i].act(s))
            ns, pi, done = env.step(acts)
            u = effective_utilities(A, pi)
            for i in range(n):
                if i not in fixed_policies:
                    agents[i].update(s, acts[i], u[i], ns, done)
            s = ns
    return [ag.Q for ag in agents]


def _eval_raw(game_name, A, horizon, Q, episodes, seed, game_kwargs=None,
              fixed_policies=None):
    """Greedy rollout; return mean raw payoff per agent and cooperation rate."""
    rng = np.random.default_rng(seed)
    base = make_game(game_name, **(game_kwargs or {}))
    env = RepeatedGame(base, horizon=horizon)
    n = env.n_agents
    fixed_policies = fixed_policies or {}
    coop_idx = base.coop_action
    tot = np.zeros(n)
    coop = 0
    steps = 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            acts = []
            for i in range(n):
                if i in fixed_policies:
                    acts.append(coop_idx if fixed_policies[i] == "cooperate"
                                else (1 - coop_idx))
                else:
                    a = int(np.argmax(Q[i][s])) if rng.random() > 0.05 \
                        else int(rng.integers(env.n_actions))
                    acts.append(a)
            ns, pi, done = env.step(acts)
            tot += pi
            coop += sum(int(a == coop_idx) for a in acts)
            steps += n
            s = ns
    return tot / episodes / max(1, horizon), coop / steps


def run_endogenous(game_name="IPD", n_agents=2, horizon=1, outer=40,
                   inner_ep=500, readapt_ep=500, eval_ep=200, eta=0.2,
                   delta=0.1, cap=0.9, seed=0, fixed_selfish=None, assort=0.0,
                   mutual=False, game_kwargs=None, out=None):
    """Two-timescale endogenous-interdependence learning loop.

    assort in [0,1]: assortative matching (Alger-Weibull). When estimating
    agent i's incentive to care about a *learner* j, the partner's reciprocal
    caring alpha_{ji} co-moves by assort*delta -- i.e. prosocial agents tend to
    meet prosocial agents. assort=0 recovers the pure self-interested gradient
    (which collapses to selfishness, cf. Dekel et al. 2007); assort=1 is full
    reciprocity. A fixed defector never reciprocates, so caring about it is never
    rewarded -> directed alpha.
    """
    fixed_selfish = set(fixed_selfish or [])          # agents with frozen alpha=0
    fixed_policies = {i: "defect" for i in fixed_selfish}
    learners = [i for i in range(n_agents) if i not in fixed_selfish]
    A = np.zeros((n_agents, n_agents))
    history = []

    def _probe(A0, i, j, sign):
        Ap = A0.copy()
        Ap[i, j] = np.clip(Ap[i, j] + sign * delta, 0.0, cap)
        if assort > 0 and j in learners:        # partner reciprocates (assortment)
            Ap[j, i] = np.clip(Ap[j, i] + sign * assort * delta, 0.0, cap)
        return _safe_A(Ap, cap)

    for t in range(outer):
        A = _safe_A(A, cap)
        Q = _train_inner(game_name, A, horizon, inner_ep, seed + t,
                         game_kwargs=game_kwargs, fixed_policies=fixed_policies)
        base_pi, coop = _eval_raw(game_name, A, horizon, Q, eval_ep, seed + 1000 + t,
                                  game_kwargs=game_kwargs, fixed_policies=fixed_policies)
        def _probe_payoff(Apert, tag):
            Qx = _train_inner(game_name, Apert, horizon, readapt_ep, seed + t,
                             game_kwargs=game_kwargs, fixed_policies=fixed_policies)
            px, _ = _eval_raw(game_name, Apert, horizon, Qx, eval_ep, seed + tag + t,
                              game_kwargs=game_kwargs, fixed_policies=fixed_policies)
            return px

        if mutual:
            # negotiated mutual caring: a shared alpha per pair climbs on the
            # JOINT raw payoff (pi_i + pi_j), enforcing symmetry (Homo-Moralis).
            for i in learners:
                for j in learners:
                    if j <= i:
                        continue
                    Ap = A.copy(); Ap[i, j] = Ap[j, i] = np.clip(A[i, j] + delta, 0, cap)
                    Ap = _safe_A(Ap, cap)
                    Am = A.copy(); Am[i, j] = Am[j, i] = max(0.0, A[i, j] - delta)
                    Am = _safe_A(Am, cap)
                    pip = _probe_payoff(Ap, 2000); pim = _probe_payoff(Am, 3000)
                    g = ((pip[i] + pip[j]) - (pim[i] + pim[j])) / (2 * delta)
                    new = float(np.clip(A[i, j] + eta * g, 0.0, cap))
                    A[i, j] = A[j, i] = new
        else:
            # self-interested gradient: each agent ascends its OWN raw payoff
            grad = np.zeros((n_agents, n_agents))
            for i in learners:
                for j in range(n_agents):
                    if j == i:
                        continue
                    pip = _probe_payoff(_probe(A, i, j, +1), 2000)
                    pim = _probe_payoff(_probe(A, i, j, -1), 3000)
                    grad[i, j] = (pip[i] - pim[i]) / (2 * delta)
            for i in learners:
                for j in range(n_agents):
                    if j != i:
                        A[i, j] = float(np.clip(A[i, j] + eta * grad[i, j], 0.0, cap))
        row = dict(t=t, coop=float(coop), welfare=float(base_pi.sum()),
                   A=[[round(float(A[i, j]), 4) for j in range(n_agents)]
                      for i in range(n_agents)],
                   raw_payoff=[round(float(x), 3) for x in base_pi])
        history.append(row)
        amean = np.mean([A[i, j] for i in learners for j in range(n_agents) if j != i])
        print(f"[endo t={t:02d}] coop={coop:.2f} welfare={base_pi.sum():.2f} "
              f"mean_alpha={amean:.3f}", flush=True)

    if out:
        outp = ROOT / out
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            for r in history:
                f.write(json.dumps(r) + "\n")
        print(f"[endo] wrote {len(history)} rows -> {outp}", flush=True)
    return history


SETTINGS = {
    # (0) naive self-interested gradient, no assortment. delta matched to the
    # other 2-player settings (0.5) so the failure/success contrast is FAIR.
    "naive2": dict(game_name="IPD", n_agents=2, horizon=1, assort=0.0, delta=0.5),
    # (a) internalize helping: 2-player self-play with assortative reciprocity
    "reciprocal2": dict(game_name="IPD", n_agents=2, horizon=1, assort=1.0, delta=0.5),
    # (b) directed: 3 agents, agent 2 is a frozen defector; do 0,1 learn to care
    # about each other (assort) but NOT about the non-reciprocating defector?
    "mixed3": dict(game_name="IPD", n_agents=3, horizon=1, fixed_selfish=[2],
                   assort=1.0, delta=0.5),
    # (c) internalize helping EACH OTHER: shared alpha negotiated on joint payoff
    # -> symmetric, stable mutual care (vs the asymmetric exploitation of naive).
    "mutual2": dict(game_name="IPD", n_agents=2, horizon=1, mutual=True, delta=0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=list(SETTINGS), default="reciprocal2")
    ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    cfg = dict(SETTINGS[args.setting])
    if cfg["n_agents"] > 2 and cfg["game_name"] == "IPD":
        cfg["game_name"] = "PGG"
        cfg["game_kwargs"] = {"n_agents": cfg["n_agents"], "mult": 1.6}
    outp = ROOT / f"results/endogenous_{args.setting}.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for s in range(args.seeds):
        print(f"=== {args.setting} seed {s} ===", flush=True)
        hist = run_endogenous(outer=args.outer, seed=s, out=None, **cfg)
        for r in hist:
            r["seed"] = s
            all_rows.append(r)
        with open(outp, "w") as f:        # write incrementally across seeds
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
    print(f"[endo] wrote {len(all_rows)} rows ({args.seeds} seeds) -> {outp}", flush=True)


if __name__ == "__main__":
    main()
