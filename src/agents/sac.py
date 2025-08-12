import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.replay_buffer import ReplayBuffer
from src.utils import epsilon_by_frame
from src.config import Config
from src.agents.base import BaseAgent


class Actor(nn.Module):
    """
    SAC Actor (Policy) Network for discrete action spaces.
    Outputs logits for a categorical policy.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gumbel-Softmax for sampling
        action_probs = F.gumbel_softmax(logits, tau=1.0, hard=True)

        # The action is the index of the max value in the one-hot vector
        action = torch.argmax(action_probs, dim=-1)

        # The log_prob is the log_prob of the chosen action
        log_prob = (probs * log_probs).sum(-1, keepdim=True)

        return action, log_prob, action_probs


class Critic(nn.Module):
    """
    SAC Critic (Q-value) Network.
    Approximates the Q-value for a state-action pair.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1 architecture
        self.net1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2 architecture
        self.net2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], 1)
        q1 = self.net1(sa)
        q2 = self.net2(sa)
        return q1, q2


class SACAgent(BaseAgent):
    @property
    def agent_type(self) -> str:
        return "sac"

    """
    Soft Actor-Critic (SAC) agent.
    """
    def __init__(
        self,
        obs_dim: int,
        action_space_n: int,
        lr: float = 3e-4, # General learning rate, can be overridden by specific ones
        gamma: float = 0.99,
        batch_size: int = 128,
        buffer_capacity: int = 100_000,
        learn_start: int = 5_000,
        tau: float = 0.005,
        alpha: float = 0.2,
        actor_lr: float = None,
        critic_lr: float = None,
        alpha_lr: float = None,
        device: str | torch.device = "auto",
        seed: Optional[int] = None,
        buffer: Optional[ReplayBuffer] = None,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.action_dim = action_space_n
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.tau = tau
        self.alpha = alpha

        self.actor = Actor(obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(obs_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        actor_lr = actor_lr if actor_lr is not None else lr
        critic_lr = critic_lr if critic_lr is not None else lr
        alpha_lr = alpha_lr if alpha_lr is not None else lr

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_entropy = -0.98 * np.log(1.0 / self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().item()

        if buffer is None:
            self.buffer = ReplayBuffer(capacity=int(buffer_capacity), obs_dim=int(obs_dim), seed=seed)
        else:
            self.buffer = buffer

        self.rng = np.random.default_rng(seed)
        self.frame_idx = 0
        self.gradient_steps = 0

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        if not isinstance(obs, np.ndarray):
            raise TypeError("obs must be a numpy.ndarray")

        if len(obs.shape) == 1:
            obs = obs[np.newaxis, :]

        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)

        with torch.no_grad():
            action, _, _ = self.actor.sample(obs_t)

        return action.item()


    def remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buffer.add(s, a, r, s2, done)

    def update(self) -> Dict[str, Any]:
        metrics = {
            "loss_critic": 0.0,
            "loss_actor": 0.0,
            "loss_alpha": 0.0,
            "alpha": self.alpha,
            "frame": int(self.frame_idx),
            "gradient_steps": int(self.gradient_steps),
        }

        if len(self.buffer) < max(self.learn_start, self.batch_size):
            return metrics

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states.astype(np.float32, copy=False)).to(self.device)
        actions_t = torch.from_numpy(actions.astype(np.int64, copy=False)).to(self.device)
        rewards_t = torch.from_numpy(rewards.astype(np.float32, copy=False)).to(self.device).unsqueeze(1)
        next_states_t = torch.from_numpy(next_states.astype(np.float32, copy=False)).to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.float32, copy=False)).to(self.device).unsqueeze(1)

        action_probs = F.one_hot(actions_t, num_classes=self.action_dim).float()

        with torch.no_grad():
            _, next_state_log_pi, next_action_probs = self.actor.sample(next_states_t)
            q1_next_target, q2_next_target = self.critic_target(next_states_t, next_action_probs)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            next_q_value = rewards_t + (1 - dones_t) * self.gamma * (min_q_next_target - self.alpha * next_state_log_pi)

        q1, q2 = self.critic(states_t, action_probs)
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        _, log_pi, action_probs_pi = self.actor.sample(states_t)
        q1_pi, q2_pi = self.critic(states_t, action_probs_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.gradient_steps += 1

        metrics["loss_critic"] = critic_loss.item()
        metrics["loss_actor"] = actor_loss.item()
        metrics["loss_alpha"] = alpha_loss.item()
        metrics["alpha"] = self.alpha
        metrics["gradient_steps"] = int(self.gradient_steps)

        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.alpha = self.log_alpha.exp().item()
