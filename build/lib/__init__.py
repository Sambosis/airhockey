"""
Self-play Air Hockey RL package.

This package organizes the environment, agents, training loop, rendering, and utilities
for a deterministic 2D air hockey game where two DQN agents learn via self-play.

Convenience exports:
- Config: Central configuration dataclass
- AirHockeyEnv, GameState, ACTIONS: Environment and related types
- QNetwork, DQNAgent: Function approximator and agent
- ReplayBuffer: Experience replay buffer
- Renderer: pygame-based renderer
- Trainer: Training orchestrator
- preprocess_obs, epsilon_by_frame, seed_everything, save_checkpoint, load_checkpoint: Utilities

Modules are imported lazily on attribute access to minimize import overhead and
avoid circular import issues during initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

__all__ = [
    "Config",
    "AirHockeyEnv",
    "GameState",
    "ACTIONS",
    "QNetwork",
    "DQNAgent",
    "ReplayBuffer",
    "Renderer",
    "Trainer",
    "preprocess_obs",
    "epsilon_by_frame",
    "seed_everything",
    "save_checkpoint",
    "load_checkpoint",
]

# Basic package metadata
__version__ = "0.1.0"

# Mapping from public symbol -> (module_path, attribute_name)
_exports: Dict[str, Tuple[str, str]] = {
    # Config
    "Config": ("src.config", "Config"),
    # Environment
    "AirHockeyEnv": ("src.env", "AirHockeyEnv"),
    "GameState": ("src.env", "GameState"),
    "ACTIONS": ("src.env", "ACTIONS"),
    # Agents
    "QNetwork": ("src.agents.dqn", "QNetwork"),
    "DQNAgent": ("src.agents.dqn", "DQNAgent"),
    # Replay buffer
    "ReplayBuffer": ("src.replay_buffer", "ReplayBuffer"),
    # Renderer
    "Renderer": ("src.render", "Renderer"),
    # Trainer
    "Trainer": ("src.train", "Trainer"),
    # Utils
    "preprocess_obs": ("src.utils", "preprocess_obs"),
    "epsilon_by_frame": ("src.utils", "epsilon_by_frame"),
    "seed_everything": ("src.utils", "seed_everything"),
    "save_checkpoint": ("src.utils", "save_checkpoint"),
    "load_checkpoint": ("src.utils", "load_checkpoint"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import and return the requested attribute from the appropriate submodule.

    This allows `from src import DQNAgent` and similar conveniences without eagerly
    importing all submodules at package import time.
    """
    if name in _exports:
        module_path, attr_name = _exports[name]
        module = import_module(module_path)
        attr = getattr(module, attr_name)
        globals()[name] = attr  # Cache for future accesses
        return attr
    raise AttributeError(f"module 'src' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


# For type checkers, import symbols eagerly so that static analysis sees them.
if TYPE_CHECKING:
    from src.config import Config as Config  # noqa: F401
    from src.env import AirHockeyEnv as AirHockeyEnv  # noqa: F401
    from src.env import GameState as GameState  # noqa: F401
    from src.env import ACTIONS as ACTIONS  # noqa: F401
    from src.agents.dqn import QNetwork as QNetwork  # noqa: F401
    from src.agents.dqn import DQNAgent as DQNAgent  # noqa: F401
    from src.replay_buffer import ReplayBuffer as ReplayBuffer  # noqa: F401
    from src.render import Renderer as Renderer  # noqa: F401
    from src.train import Trainer as Trainer  # noqa: F401
    from src.utils import preprocess_obs as preprocess_obs  # noqa: F401
    from src.utils import epsilon_by_frame as epsilon_by_frame  # noqa: F401
    from src.utils import seed_everything as seed_everything  # noqa: F401
    from src.utils import save_checkpoint as save_checkpoint  # noqa: F401
    from src.utils import load_checkpoint as load_checkpoint  # noqa: F401