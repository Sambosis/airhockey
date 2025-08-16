from __future__ import annotations

import argparse
import random
from dataclasses import replace
from typing import Dict, Tuple

from src.config import Config
from src.env import AirHockeyEnv
from src.render import Renderer
from src.agents.dqn import DQNAgent
from src.train import Trainer


def _build_agents(cfg: Config) -> tuple[DQNAgent, DQNAgent]:
    """Construct a pair of agents using values from the config."""
    kwargs = dict(
        obs_dim=cfg.obs_dim,
        action_space_n=cfg.action_space_n,
        lr=cfg.lr,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        buffer_capacity=cfg.buffer_capacity,
        learn_start=cfg.learn_start,
        target_sync=cfg.target_sync,
        eps_start=cfg.eps_start,
        eps_end=cfg.eps_end,
        eps_decay_frames=cfg.eps_decay_frames,
        device=cfg.device_str,
    )
    return DQNAgent(**kwargs), DQNAgent(**kwargs)


def evaluate(cfg: Config, episodes: int) -> float:
    """Run a short training session and return mean left-agent reward."""
    env = AirHockeyEnv(cfg)
    left, right = _build_agents(cfg)
    renderer = Renderer(width=cfg.width, height=cfg.height, fps=cfg.render_fps)
    trainer = Trainer(
        env=env,
        agent_left=left,
        agent_right=right,
        renderer=renderer,
        visualize_every_n=0,
        device=cfg.device_str,
        checkpoint_every=0,
    )
    trainer.run(episodes)
    return float(sum(trainer.recent_returns_left) / len(trainer.recent_returns_left))


def random_search(base: Config, ranges: Dict[str, Tuple[float, float]], *, trials: int, episodes: int) -> Config:
    """Randomly sample hyperparameters within ranges and keep the best."""
    best_cfg = base
    best_reward = float("-inf")
    for _ in range(trials):
        sampled = {}
        for name, (low, high) in ranges.items():
            if isinstance(low, int) and isinstance(high, int):
                sampled[name] = random.randint(low, high)
            else:
                sampled[name] = random.uniform(float(low), float(high))
        cfg = replace(base, **sampled)
        reward = evaluate(cfg, episodes)
        if reward > best_reward:
            best_reward, best_cfg = reward, cfg
    return best_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Random hyperparameter search")
    parser.add_argument("--trials", type=int, default=3, help="Number of random trials")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per trial")
    args = parser.parse_args()

    base_cfg = Config(episodes=args.episodes)
    ranges = {
        "lr": (1e-5, 5e-4),
        "batch_size": (64, 512),
        "eps_decay_frames": (100_000, 1_000_000),
    }
    best = random_search(base_cfg, ranges, trials=args.trials, episodes=args.episodes)
    print("Best config:", best)


if __name__ == "__main__":
    main()
