import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.replay_buffer import ReplayBuffer
from src.utils import epsilon_by_frame

__all__ = ["DuelingQNetwork", "DuelingDQNAgent"]


class DuelingQNetwork(nn.Module):
    """
    MLP with dueling architecture for discrete action-value estimation.

    - Shared feature extractor
    - Separate value and advantage streams
    - Aggregation: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """

    def __init__(self, obs_dim: int, action_space_n: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space_n = action_space_n

        hidden = 512
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_space_n),
        )

        # Weight initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        value = self.value_head(h)              # (B, 1)
        advantage = self.adv_head(h)            # (B, A)
        advantage_centered = advantage - advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage_centered   # broadcast (B, 1) + (B, A)
        return q_values


class DuelingDQNAgent:
    """
    Dueling Double DQN agent with a target network, replay buffer, and epsilon-greedy policy.
    Matches the public API of DQNAgent for compatibility with Trainer and checkpoint utils.
    """

    def __init__(
        self,
        obs_dim: int,
        action_space_n: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        epsilon_start: float | None = None,
        epsilon_end: float | None = None,
        epsilon_decay_steps: int | None = None,
        # alias names to match config/main
        eps_start: float | None = None,
        eps_end: float | None = None,
        eps_decay_frames: int | None = None,
        target_sync: int = 1_000,
        learn_start: int = 5_000,
        buffer: Optional[ReplayBuffer] = None,
        buffer_capacity: int = 100_000,
        device: str | torch.device = "auto",
        seed: Optional[int] = None,
    ) -> None:
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.action_space_n = action_space_n

        # Replay buffer
        if buffer is None:
            self.buffer = ReplayBuffer(capacity=int(buffer_capacity), obs_dim=int(obs_dim), seed=seed)
        else:
            self.buffer = buffer

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learn_start = int(learn_start)
        self.target_sync = int(target_sync)

        # Support both naming conventions for epsilon schedule
        e_start = eps_start if eps_start is not None else epsilon_start
        e_end = eps_end if eps_end is not None else epsilon_end
        e_decay = eps_decay_frames if eps_decay_frames is not None else epsilon_decay_steps

        self.eps_start = float(1.0 if e_start is None else e_start)
        self.epsilon_min = float(0.05 if e_end is None else e_end)
        self.epsilon_decay_steps = int(100_000 if e_decay is None else e_decay)
        self.epsilon = self.eps_start

        self.q_net = DuelingQNetwork(obs_dim, action_space_n).to(self.device)
        self.target_net = DuelingQNetwork(obs_dim, action_space_n).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Tracking counters
        self.frame_idx: int = 0
        self.gradient_steps: int = 0

        # Local RNG
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        if not isinstance(obs, np.ndarray):
            raise TypeError("obs must be a numpy.ndarray")

        if training:
            self.epsilon = epsilon_by_frame(
                self.frame_idx, self.eps_start, self.epsilon_min, self.epsilon_decay_steps
            )
        else:
            self.epsilon = 0.0

        self.frame_idx += 1

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_space_n))

        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)
            q_values = self.q_net(obs_t.unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buffer.add(s, int(a), float(r), s2, bool(done))

    def update(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "loss": 0.0,
            "epsilon": float(self.epsilon),
            "frame": int(self.frame_idx),
            "gradient_steps": int(self.gradient_steps),
        }
        if len(self.buffer) < max(self.learn_start, self.batch_size):
            return metrics

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states.astype(np.float32, copy=False)).to(self.device)
        next_states_t = torch.from_numpy(next_states.astype(np.float32, copy=False)).to(self.device)
        actions_t = torch.from_numpy(actions.astype(np.int64, copy=False)).to(self.device)
        rewards_t = torch.from_numpy(rewards.astype(np.float32, copy=False)).to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.float32, copy=False)).to(self.device)

        # Current Q-values
        q_values = self.q_net(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target using dueling target network
        with torch.no_grad():
            next_q_online = self.q_net(next_states_t)
            best_next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(next_states_t)
            next_q_selected = next_q_target.gather(1, best_next_actions).squeeze(1)
            targets = rewards_t + (1.0 - dones_t) * self.gamma * next_q_selected

        loss = F.smooth_l1_loss(q_selected, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.target_sync == 0:
            self.sync_target()

        metrics["loss"] = float(loss.item())
        metrics["epsilon"] = float(self.epsilon)
        metrics["frame"] = int(self.frame_idx)
        metrics["gradient_steps"] = int(self.gradient_steps)
        return metrics

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "meta": {
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "eps_start": self.eps_start,
                "epsilon_decay_steps": self.epsilon_decay_steps,
                "gamma": self.gamma,
                "obs_dim": self.obs_dim,
                "action_space_n": self.action_space_n,
                "batch_size": self.batch_size,
                "learn_start": self.learn_start,
                "target_sync": self.target_sync,
                "frame_idx": self.frame_idx,
                "gradient_steps": self.gradient_steps,
                "device": str(self.device),
            },
        }
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[str | torch.device] = None) -> None:
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        self.q_net.load_state_dict(checkpoint["q_net"]) 
        self.target_net.load_state_dict(checkpoint["target_net"]) 
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"]) 
        meta: Dict[str, Any] = checkpoint.get("meta", {})
        self.epsilon = float(meta.get("epsilon", self.epsilon))
        self.epsilon_min = float(meta.get("epsilon_min", self.epsilon_min))
        self.eps_start = float(meta.get("eps_start", self.eps_start))
        self.epsilon_decay_steps = int(meta.get("epsilon_decay_steps", self.epsilon_decay_steps))
        self.gamma = float(meta.get("gamma", self.gamma))
        self.batch_size = int(meta.get("batch_size", self.batch_size))
        self.learn_start = int(meta.get("learn_start", self.learn_start))
        self.target_sync = int(meta.get("target_sync", self.target_sync))
        self.frame_idx = int(meta.get("frame_idx", self.frame_idx))
        self.gradient_steps = int(meta.get("gradient_steps", self.gradient_steps))