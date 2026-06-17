from .qlearning import TabularQLearner

__all__ = ["TabularQLearner", "DQNLearner"]


def __getattr__(name):
    # lazy import so the tabular/CPU path never imports torch
    if name == "DQNLearner":
        from .dqn import DQNLearner
        return DQNLearner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
