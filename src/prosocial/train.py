"""Self-play IQL training loop on repeated matrix games under a reward transform.

The environment yields RAW payoffs pi; the transform maps pi -> effective
rewards the agents learn on. Metrics are always computed on RAW payoffs pi
(social welfare, cooperation) so conditions with different transforms are
compared on the same material footing (plan.md Exp 1 metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .envs import RepeatedGame, make_game
from .rewards import RewardTransform

# default learning rate per learner ("lr" left at the tabular default for
# backward compatibility; SGD on the DQN net wants a much smaller step).
_DEFAULT_LR = {"tabular": 0.1, "dqn": 1e-3}


@dataclass
class TrainResult:
    coop_rate: float          # fraction cooperative actions over last `eval_frac`
    coop_stability: float     # std of per-episode coop rate over last `eval_frac`
    social_welfare: float     # mean total RAW payoff per round over last `eval_frac`
    gini: float               # gini of cumulative RAW payoff over last `eval_frac`
    coop_curve: np.ndarray = field(repr=False, default=None)


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def train_selfplay(game_name: str, transform: RewardTransform, horizon=1,
                   episodes=4000, lr=None, gamma=0.9, seed=0,
                   eval_frac=0.1, game_kwargs=None, agent_kwargs=None,
                   learner="tabular", device="cpu") -> TrainResult:
    """Self-play loop. `learner="tabular"` (default) uses the numpy IQL table;
    `learner="dqn"` swaps in the torch deep-Q learner (`device="cuda"` to run on
    GPU). Both expose the same act/update/set_epsilon API, so the loop below is
    identical for either."""
    rng = np.random.default_rng(seed)
    base = make_game(game_name, **(game_kwargs or {}))
    env = RepeatedGame(base, horizon=horizon)
    n = env.n_agents
    if lr is None:
        lr = _DEFAULT_LR[learner]
    if learner == "tabular":
        from .agents import TabularQLearner as AgentCls
        akw = dict(lr=lr, gamma=gamma, rng=rng)
    elif learner == "dqn":
        from .agents import DQNLearner as AgentCls
        akw = dict(lr=lr, gamma=gamma, rng=rng, device=device)
    else:
        raise ValueError(f"unknown learner {learner!r}")
    akw.update(agent_kwargs or {})
    agents = [AgentCls(env.n_states, env.n_actions, **akw) for _ in range(n)]
    coop_idx = base.coop_action

    coop_curve = np.zeros(episodes)
    welfare_curve = np.zeros(episodes)
    payoff_accum = np.zeros(n)

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        for ag in agents:
            ag.set_epsilon(frac)
        state = env.reset()
        ep_coop = 0
        ep_rounds = 0
        ep_welfare = 0.0
        done = False
        while not done:
            actions = [agents[i].act(state) for i in range(n)]
            next_state, pi, done = env.step(actions)
            r = transform(pi)  # learn on transformed reward
            for i in range(n):
                agents[i].update(state, actions[i], r[i], next_state, done)
            state = next_state
            ep_coop += sum(int(a == coop_idx) for a in actions)
            ep_rounds += n
            ep_welfare += pi.sum()
            payoff_accum += pi
        coop_curve[ep] = ep_coop / ep_rounds
        welfare_curve[ep] = ep_welfare / max(1, env.horizon)

    k = max(1, int(episodes * eval_frac))
    tail_coop = coop_curve[-k:]
    return TrainResult(
        coop_rate=float(tail_coop.mean()),
        coop_stability=float(tail_coop.std()),
        social_welfare=float(welfare_curve[-k:].mean()),
        gini=_gini(payoff_accum),
        coop_curve=coop_curve,
    )
