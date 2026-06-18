"""Independent PPO (clipped) actor-critic, for the algorithm comparison (M3).

A minimal on-policy PPO with the clipped surrogate objective and a value
baseline, sharing the small MLP body used by the A2C learner. It buffers
transitions and runs several epochs of minibatch updates once enough are
collected, which is the stabilising difference from plain A2C.

Two front-ends so it serves both tracks:
  * matrix / bandit one-shot: ``act_state(idx)`` one-hot encodes a discrete state
    index (drop-in for the IQL/DQN matrix loop), ``store`` + ``maybe_update``.
  * spatial: ``act(obs)`` consumes a float observation vector and the
    episode-buffered ``store_spatial`` + ``finish_episode`` mirror A2CAgent.

Cooperation is always learned on the *transformed* reward, like every other
learner; PPO only changes the policy-optimisation rule.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _PVNet(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class PPOLearner:
    """PPO with a discrete-state (one-hot) front-end matching IQL/DQN's API."""

    def __init__(self, n_states, n_actions, lr=3e-3, gamma=0.9,
                 eps_start=0.5, eps_end=0.02, rng=None, device="cpu",
                 hidden=64, clip=0.2, epochs=4, batch=256, ent_coef=0.01,
                 onehot=True):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps = eps_start                  # only used to mirror the API
        self.rng = rng or np.random.default_rng()
        self.device = torch.device(device)
        torch.manual_seed(int(self.rng.integers(2**31 - 1)))
        self.onehot = onehot
        in_dim = n_states if onehot else n_states
        self.net = _PVNet(in_dim, n_actions, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip = clip
        self.epochs = epochs
        self.batch = batch
        self.ent_coef = ent_coef
        self._eye = torch.eye(n_states, device=self.device)
        self._buf = []   # (state_idx, action, logp, reward)

    def _encode(self, idx):
        return self._eye[torch.as_tensor(idx, dtype=torch.long, device=self.device)]

    @torch.no_grad()
    def act(self, state, greedy=False):
        x = self._encode([state])
        logits, _ = self.net(x)
        dist = torch.distributions.Categorical(logits=logits[0])
        if greedy:
            a = int(torch.argmax(logits[0]))
        else:
            a = int(dist.sample())
        self._last_logp = float(dist.log_prob(torch.tensor(a, device=self.device)))
        return a

    def update(self, s, a, r, s_next, done):
        """Store the one-shot transition; PPO update fires when the batch fills."""
        self._buf.append((s, a, self._last_logp, float(r)))
        if len(self._buf) >= self.batch:
            self._ppo_update()

    def _ppo_update(self):
        if not self._buf:
            return
        states = self._encode([b[0] for b in self._buf])
        actions = torch.as_tensor([b[1] for b in self._buf], dtype=torch.long,
                                  device=self.device)
        old_logp = torch.as_tensor([b[2] for b in self._buf], dtype=torch.float32,
                                   device=self.device)
        # one-shot (horizon=1): return is just the immediate reward
        returns = torch.as_tensor([b[3] for b in self._buf], dtype=torch.float32,
                                  device=self.device)
        for _ in range(self.epochs):
            logits, values = self.net(states)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions)
            adv = (returns - values.detach())
            if adv.numel() > 1 and adv.std() > 1e-6:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(logp - old_logp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)
            ent = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - self.ent_coef * ent
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
        self._buf = []

    def set_epsilon(self, frac_done):
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * frac_done

    def flush(self):
        """Force an update on any buffered transitions (call at end of training)."""
        self._ppo_update()
