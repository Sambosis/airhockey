from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import torch


class BaseAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents.
    """

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """
        Returns the type of the agent (e.g., 'dqn', 'sac').
        """
        pass

    @abstractmethod
    def select_action(self, obs: np.ndarray, training: bool = True) -> Any:
        """
        Select an action based on the observation.
        """
        pass

    @abstractmethod
    def remember(self, s: np.ndarray, a: Any, r: float, s2: np.ndarray, done: bool) -> None:
        """
        Store an experience tuple in the replay buffer.
        """
        pass

    @abstractmethod
    def update(self) -> Dict[str, Any]:
        """
        Perform a single learning update.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        """
        pass

    @abstractmethod
    def load(self, path: str, map_location: Optional[str | torch.device] = None) -> None:
        """
        Load the agent's state.
        """
        pass
