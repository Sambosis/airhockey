"""
Agents package.

This package provides reinforcement learning agent implementations used in the
self-play Air Hockey project. Currently, it exposes:

- QNetwork: PyTorch MLP for approximating Q-values.
- DQNAgent: Double DQN agent with target network and experience replay.
"""

from .dqn import QNetwork, DQNAgent

__all__ = ["QNetwork", "DQNAgent"]