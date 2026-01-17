"""
Agents package.

This package provides reinforcement learning agent implementations used in the
self-play Air Hockey project. Currently, it exposes:

- QNetwork: PyTorch MLP for approximating Q-values.
- DQNAgent: Double DQN agent with target network and experience replay.
- DuelingDQNAgent: Dueling network variant of Double DQN.
"""

from .dqn import QNetwork, DQNAgent
from .dueling_dqn import DuelingQNetwork, DuelingDQNAgent

__all__ = ["QNetwork", "DQNAgent", "DuelingQNetwork", "DuelingDQNAgent"]