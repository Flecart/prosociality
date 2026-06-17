"""Independent deep Q-learning (DQN) for matrix / repeated matrix games.

A torch reimplementation of the tabular learner in `qlearning.py`: instead of a
Q[state, action] table updated by a Bellman backup, each agent holds a small MLP
Q-network trained by gradient descent on the TD error. The discrete state index
is one-hot encoded as the network input, so for the small matrix games this MLP
can represent exactly the same Q-function as the table -- but the update path is
now SGD, which is what lets it run on the GPU and scale past a table.

It carries the two classic deep-Q upgrades and *keeps both*:
  - experience replay  (decorrelate consecutive transitions)
  - a target network    (stabilise the bootstrap target)

The public API mirrors `TabularQLearner` exactly -- `act(state, greedy)`,
`update(s, a, r, s_next, done)`, `set_epsilon(frac_done)` -- so it is a drop-in
agent for `train.train_selfplay` (pass `learner="dqn"`). The tabular learner is
kept alongside it; this does not replace it.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNLearner:
    def __init__(self, n_states, n_actions, lr=1e-3, gamma=0.9,
                 eps_start=0.5, eps_end=0.02, rng=None,
                 policy="egreedy", temperature=0.5,
                 device=None, hidden=64, buffer_size=10_000,
                 batch_size=64, min_buffer=128, target_sync=200):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps = eps_start
        self.rng = rng or np.random.default_rng()
        self.policy = policy          # "egreedy" | "boltzmann"
        self.temperature = temperature

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # seed torch's sampler from the shared numpy rng for reproducibility
        torch.manual_seed(int(self.rng.integers(2**31 - 1)))

        self.q = QNet(n_states, n_actions, hidden).to(self.device)
        self.target = QNet(n_states, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.target_sync = target_sync
        self._steps = 0

        # cached identity matrix for fast one-hot encoding of state indices
        self._eye = torch.eye(n_states, device=self.device)

    def _encode(self, states):
        """int index (or array of indices) -> one-hot float rows on device."""
        idx = torch.as_tensor(states, dtype=torch.long, device=self.device)
        return self._eye[idx]

    @torch.no_grad()
    def act(self, state, greedy=False):
        x = self._encode([state])
        q = self.q(x)[0]
        if self.policy == "boltzmann" and not greedy:
            p = F.softmax(q / max(1e-6, self.temperature), dim=-1)
            return int(self.rng.choice(self.n_actions, p=p.cpu().numpy()))
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        qv = q.cpu().numpy()
        best = np.flatnonzero(qv == qv.max())   # random tie-break among argmax
        return int(self.rng.choice(best))

    def update(self, s, a, r, s_next, done):
        """Store the transition and take one SGD step on a replay minibatch."""
        self.buffer.append((s, a, float(r), s_next, bool(done)))
        if len(self.buffer) < self.min_buffer:
            return
        idx = self.rng.integers(len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idx]
        s_b, a_b, r_b, sn_b, d_b = zip(*batch)

        states = self._encode(s_b)
        next_states = self._encode(sn_b)
        actions = torch.as_tensor(a_b, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(r_b, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(d_b, dtype=torch.float32, device=self.device)

        q_sa = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(next_states).max(dim=1).values
            target = rewards + (1.0 - dones) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self._steps += 1
        if self._steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())

    def set_epsilon(self, frac_done: float):
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * frac_done
