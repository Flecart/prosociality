"""Independent advantage actor-critic (A2C) for the spatial dilemmas.

Each agent has its own small MLP policy+value head and learns on its
*transformed* per-step reward U_i = [(I-A)^{-1} pi]_i. This is the temporally
extended analogue of the matrix-game IQL setup: same interdependence wrapper,
now over a gridworld with movement and regrowth dynamics. Deliberately small
and CPU-friendly (a smoke-scale learner, not a tuned PPO baseline).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class A2CAgent:
    def __init__(self, obs_dim, n_actions, lr=3e-3, gamma=0.99, device="cpu"):
        self.net = PolicyValueNet(obs_dim, n_actions).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device

    def act(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, value = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a), dist.log_prob(a), value, dist.entropy()

    def learn(self, log_probs, values, entropies, rewards):
        # n-step discounted returns
        returns, R = [], 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        adv = returns - values.detach()
        if adv.numel() > 1 and adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(log_probs * adv).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_bonus = entropies.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return float(loss.item())
