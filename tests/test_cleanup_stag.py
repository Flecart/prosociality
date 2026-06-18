"""Tests for the CleanupStag log-hunt env (M1).

Verifies the defining stag-hunt property: a log pays each lifter ONLY if
>= min_log_lifters co-lift the same log the same step, and that the env records
who co-lifted with whom (the partner-specific signal for the endogenous-A loop).
"""

import numpy as np

from prosocial.envs.spatial import CleanupStag


def _place(env, i, cell):
    env.pos[i] = np.array(cell)


def test_contract_shapes():
    env = CleanupStag(n_agents=4, size=5)
    obs = env.reset()
    assert len(obs) == env.n_agents
    assert all(o.shape == (env.obs_dim,) for o in obs)
    obs, pi, done = env.step([0, 0, 0, 0])
    assert pi.shape == (env.n_agents,)
    assert isinstance(done, bool) or isinstance(done, (np.bool_,))


def test_solo_lift_is_wasted():
    env = CleanupStag(n_agents=4, size=5, log_cells=[(1, 1)], min_log_lifters=2)
    env.reset()
    env.logs[(1, 1)] = True
    _place(env, 0, (1, 1))           # only one agent at the log
    for i in (1, 2, 3):
        _place(env, i, (4, 4))
    actions = [env.LIFT, 0, 0, 0]
    _, pi, _ = env.step(actions)
    assert pi[0] == 0.0               # lifting alone pays nothing
    assert env.last_info["joint_lifts"] == 0
    assert env.last_info["wasted_lifts"] == 1


def test_joint_lift_pays_each_and_records_pair():
    env = CleanupStag(n_agents=4, size=5, log_cells=[(1, 1)],
                      min_log_lifters=2, log_reward=5.0)
    env.reset()
    env.logs[(1, 1)] = True
    _place(env, 0, (1, 1))
    _place(env, 1, (1, 2))           # Chebyshev-adjacent -> can co-lift
    for i in (2, 3):
        _place(env, i, (4, 4))
    _, pi, _ = env.step([env.LIFT, env.LIFT, 0, 0])
    assert pi[0] == 5.0 and pi[1] == 5.0   # each lifter paid
    assert pi[2] == 0.0 and pi[3] == 0.0
    assert env.last_info["joint_lifts"] == 1
    assert env.pair_lifts[0, 1] == 1 and env.pair_lifts[1, 0] == 1
    assert (0, 1) in env.step_colift_pairs
    assert env.logs[(1, 1)] is False       # log consumed


def test_three_way_lift_credits_all_pairs():
    env = CleanupStag(n_agents=4, size=5, log_cells=[(2, 2)], min_log_lifters=2)
    env.reset()
    env.logs[(2, 2)] = True
    for i, cell in zip(range(3), [(2, 2), (2, 3), (3, 2)]):
        _place(env, i, cell)
    _place(env, 3, (0, 0))
    _, pi, _ = env.step([env.LIFT, env.LIFT, env.LIFT, 0])
    assert (pi[:3] == 5.0).all() and pi[3] == 0.0
    # all 3 pairs among {0,1,2} recorded
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert env.pair_lifts[a, b] == 1


def test_harvest_apple_is_solo():
    env = CleanupStag(n_agents=2, size=3, log_cells=[(0, 0)])
    env.reset()
    env.apples[1, 1] = 1.0
    _place(env, 0, (1, 1))
    _place(env, 1, (2, 2))
    _, pi, _ = env.step([5, 0])      # agent 0 harvests
    assert pi[0] == env.apple_reward
    assert env.last_info["harvest"] == 1


def test_coop_signal_and_joint_rate():
    env = CleanupStag(n_agents=2, size=3, log_cells=[(0, 0)], min_log_lifters=2)
    env.reset()
    # one joint lift, one apple -> 2 lift credits vs 1 harvest
    env.last_info["lift_credits"] = 2
    env.last_info["harvest"] = 1
    assert abs(env.coop_signal() - 2 / 3) < 1e-9
    env.last_info["joint_lifts"] = 3
    env.t = 30
    assert abs(env.joint_lift_rate() - 0.1) < 1e-9
