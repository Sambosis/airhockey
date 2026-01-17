import numpy as np
from typing import Tuple, Union, Optional


class ReplayBuffer:
    """
    Fixed-size cyclic experience replay buffer for off-policy RL.

    Stores transitions (state, action, reward, next_state, done) in pre-allocated
    NumPy arrays to minimize per-step allocations and enable fast random sampling.

    Attributes:
        capacity: Maximum number of transitions stored.
        obs_dim: Dimensionality of the observation vector.
        ptr: Insertion pointer (next index to write).
        size: Current number of stored transitions (<= capacity).
        states: Array of states with shape (capacity, obs_dim), dtype float32.
        actions: Array of actions with shape (capacity,), dtype int64.
        rewards: Array of rewards with shape (capacity,), dtype float32.
        next_states: Array of next states with shape (capacity, obs_dim), dtype float32.
        dones: Array of done flags with shape (capacity,), dtype bool.
        rng: NumPy random Generator used for sampling.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store. Must be > 0.
            obs_dim: Size of the observation vector. Must be > 0.
            seed: Optional random seed or an existing np.random.Generator for sampling.

        Raises:
            ValueError: If capacity or obs_dim are not positive.
        """
        if capacity is None or capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be a positive integer.")
        if obs_dim is None or obs_dim <= 0:
            raise ValueError("ReplayBuffer obs_dim must be a positive integer.")

        self.capacity: int = int(capacity)
        self.obs_dim: int = int(obs_dim)

        # Pre-allocate storage
        self.states: np.ndarray = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards: np.ndarray = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states: np.ndarray = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones: np.ndarray = np.zeros((self.capacity,), dtype=np.bool_)

        self.ptr: int = 0
        self.size: int = 0

        if isinstance(seed, np.random.Generator):
            self.rng: np.random.Generator = seed
        else:
            self.rng = np.random.default_rng(seed)

    def add(
        self,
        s: np.ndarray,
        a: Union[int, np.integer],
        r: Union[float, np.floating],
        s2: np.ndarray,
        done: Union[bool, np.bool_],
    ) -> None:
        """
        Add a transition to the buffer, overwriting the oldest if full.

        Args:
            s: State vector, shape (obs_dim,). Values will be cast to float32.
            a: Action taken (integer).
            r: Reward received (float).
            s2: Next state vector, shape (obs_dim,). Values cast to float32.
            done: Episode termination flag.

        Raises:
            ValueError: If input state shapes do not match obs_dim.
        """
        if s is None or s2 is None:
            raise ValueError("States s and s2 must not be None.")
        if np.shape(s)[-1] != self.obs_dim or np.shape(s2)[-1] != self.obs_dim:
            raise ValueError(
                f"State shapes must be (*, {self.obs_dim}). "
                f"Got {np.shape(s)} and {np.shape(s2)}"
            )

        idx = self.ptr

        # Assign with dtype casting handled by NumPy on assignment
        self.states[idx, ...] = np.asarray(s, dtype=np.float32)
        self.actions[idx] = int(a)
        self.rewards[idx] = float(r)
        self.next_states[idx, ...] = np.asarray(s2, dtype=np.float32)
        self.dones[idx] = bool(done)

        # Advance pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Uniformly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) where:
              - states: float32 array, shape (batch_size, obs_dim)
              - actions: int64 array, shape (batch_size,)
              - rewards: float32 array, shape (batch_size,)
              - next_states: float32 array, shape (batch_size, obs_dim)
              - dones: float32 array (0.0 or 1.0), shape (batch_size,)

        Raises:
            ValueError: If batch_size is invalid or there are not enough samples.
        """
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.size < batch_size:
            raise ValueError(
                f"Not enough samples to draw from buffer: size={self.size}, requested={batch_size}"
            )

        idxs = self.rng.integers(0, self.size, size=batch_size, endpoint=False)

        s = self.states[idxs]
        a = self.actions[idxs]
        r = self.rewards[idxs]
        s2 = self.next_states[idxs]
        d = self.dones[idxs].astype(np.float32)  # convert to 0.0/1.0 for convenience

        return s, a, r, s2, d

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.size