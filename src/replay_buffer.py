import numpy as np
from typing import Tuple, Union, Optional
import random


class ReplayBuffer:
    """
    Fixed-size cyclic experience replay buffer for off-policy RL.

    Stores transitions (state, action, reward, next_state, done) in pre-allocated
    NumPy arrays to minimize per-step allocations and enable fast random sampling.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
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
        return self.size


class SumTree:
    """
    Binary tree for efficient prioritized sampling.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    """
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, seed: Optional[int] = None):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priorities
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        experience = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, experience)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                              np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Increase beta over time
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # Compute importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        # Convert to arrays
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        return states, actions, rewards, next_states, dones, np.array(idxs), is_weights

    def update_priorities(self, idxs: np.ndarray, errors: np.ndarray):
        for idx, error in zip(idxs, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries