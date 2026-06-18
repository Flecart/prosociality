from .qlearning import TabularQLearner

__all__ = ["TabularQLearner", "DQNLearner", "PPOLearner", "A2CAgent"]


def __getattr__(name):
    # lazy import so the tabular/CPU path never imports torch
    if name == "DQNLearner":
        from .dqn import DQNLearner
        return DQNLearner
    if name == "PPOLearner":
        from .ppo import PPOLearner
        return PPOLearner
    if name == "A2CAgent":
        from .a2c import A2CAgent
        return A2CAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
