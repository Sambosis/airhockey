from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Internal imports
from src.env import AirHockeyEnv, GameState
from src.render import Renderer
from src.agents.dqn import DQNAgent
from src.utils import save_checkpoint
from src.config import Config
from recorder import record_pygame



class Trainer:
    """
    Orchestrates self-play training between two DQN agents in the AirHockeyEnv.

    Responsibilities:
    - Episode loop: reset env, interact until terminal, store transitions, update agents.
    - Rendering cadence: open pygame display only on visualize episodes; avoid pygame otherwise.
    - Logging: maintain simple running averages and print concise summaries.
    - Checkpointing: periodically saves both agents' states and training metadata.

    Notes:
    - Observations from the environment are already normalized; directly stored in replay buffers.
    - One update per environment step is attempted when enough samples exist.
    - Right agent receives a mirrored observation from the environment for symmetry.
    """

    def __init__(
        self,
        env: AirHockeyEnv,
        agent_left: DQNAgent,
        agent_right: DQNAgent,
        renderer: Renderer,
        visualize_every_n: int = 5,
        device: str | None = None,
        checkpoint_dir: str = "checkpoints",
        checkpoint_every: int = 500,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.agent_left = agent_left
        self.agent_right = agent_right
        self.renderer = renderer

        self.visualize_every_n = max(0, int(visualize_every_n))
        self.device = device or "cpu"
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = max(0, int(checkpoint_every))

        # Episode counter (1-based for human-friendly logging)
        self.episode: int = 0

        # Stats
        self.total_env_steps: int = 0
        self.recent_returns_left: deque[float] = deque(maxlen=100)
        self.recent_returns_right: deque[float] = deque(maxlen=100)
        self.recent_lengths: deque[int] = deque(maxlen=100)

        # Local RNG if needed
        self.rng = np.random.default_rng(seed)

    def run(self, num_episodes: int) -> None:
        """
        Run a multi-episode training loop.

        Args:
            num_episodes: Number of episodes to run.
        """
        num_episodes = int(num_episodes)
        if num_episodes <= 0:
            raise ValueError("num_episodes must be > 0")

        start_time = time.time()
        for i in range(num_episodes):
            self.episode += 1
            ep_index = self.episode  # 1-based
            visualize = self.visualize_every_n > 0 and (ep_index % self.visualize_every_n == 0)

            ep_start = time.time()
            result = self.run_episode(visualize=visualize)
            ep_dur = time.time() - ep_start

            # Update rolling stats
            self.recent_returns_left.append(result["total_reward_left"])
            self.recent_returns_right.append(result["total_reward_right"])
            self.recent_lengths.append(result["steps"])

            avg_len = float(np.mean(self.recent_lengths)) if self.recent_lengths else result["steps"]
            avg_ret_l = float(np.mean(self.recent_returns_left)) if self.recent_returns_left else result["total_reward_left"]
            avg_ret_r = float(np.mean(self.recent_returns_right)) if self.recent_returns_right else result["total_reward_right"]

            # Print concise episode summary
            print(
                f"[Episode {ep_index:6d}] steps={result['steps']:4d} "
                f"goal={result['goal'] or 'none':>5s} "
                f"R_left={result['total_reward_left']:+.3f} "
                f"R_right={result['total_reward_right']:+.3f} "
                f"eps(L/R)=({result['epsilon_left']:.3f}/{result['epsilon_right']:.3f}) "
                f"avg_len={avg_len:.1f} "
                f"avg_R(L/R)=({avg_ret_l:+.3f}/{avg_ret_r:+.3f}) "
                f"{'(viz)' if visualize else ''} "
                f"dur={ep_dur:.2f}s"
            )

            # Maybe checkpoint
            try:
                self.maybe_checkpoint()
            except Exception as e:
                print(f"[Trainer] Warning: checkpointing failed on episode {self.episode}: {e}")

        total_dur = time.time() - start_time
        print(f"[Trainer] Finished {num_episodes} episodes in {total_dur/60.0:.2f} min. Total env steps: {self.total_env_steps}")

    def run_episode(self, visualize: bool) -> Dict[str, Any]:
        """
        Run a single self-play episode.

        Args:
            visualize: If True, open the renderer and draw each frame at configured FPS.
                       If False, avoid any pygame calls.

        Returns:
            dict with episode metrics: steps, total_reward_left/right, goal, score_left/right,
            epsilon_left/right, avg_loss_left/right.
        """
        # Only open the renderer for visualize episodes
        if visualize:
            try:
                self.renderer.open()
            except Exception as e:
                print(f"[Trainer] Renderer.open() failed: {e}")
                visualize = False  # Disable visualization for this episode

        obs_left, obs_right = self.env.reset()  # Already normalized; right is mirrored
        done = False

        steps = 0
        total_reward_left = 0.0
        total_reward_right = 0.0

        # Track average loss over the episode
        loss_sum_left = 0.0
        loss_sum_right = 0.0
        loss_count_left = 0
        loss_count_right = 0

        last_info: Dict[str, Any] = {}
        with record_pygame(f"videos/replay_hockey_{self.episode}.mp4", fps=60) as rec:
            while not done:
                # Agents select actions (epsilon-greedy handled internally)
                a_left = self.agent_left.select_action(obs_left, training=True)
                a_right = self.agent_right.select_action(obs_right, training=True)

                # Environment step
                nobs_left, nobs_right, r_left, r_right, done, info = self.env.step(a_left, a_right)
                steps += 1
                self.total_env_steps += 1

                # Store transitions
                self.agent_left.remember(obs_left, a_left, r_left, nobs_left, done)
                self.agent_right.remember(obs_right, a_right, r_right, nobs_right, done)

                # Online update (one step per env tick if enough samples)
                metrics_left = self.agent_left.update()
                metrics_right = self.agent_right.update()

                # Accumulate losses if available
                if metrics_left and "loss" in metrics_left:
                    loss_sum_left += float(metrics_left["loss"])
                    loss_count_left += 1
                if metrics_right and "loss" in metrics_right:
                    loss_sum_right += float(metrics_right["loss"])
                    loss_count_right += 1

                # Rewards accumulation
                total_reward_left += r_left
                total_reward_right += r_right

                # Rendering (only on visualize episodes)
                if visualize and "state" in info and isinstance(info["state"], GameState):
                    try:
                        self.renderer.draw(
                            state=info["state"],
                            episode_idx=self.episode,
                            eps_left=float(getattr(self.agent_left, "epsilon", 0.0)),
                            eps_right=float(getattr(self.agent_right, "epsilon", 0.0)),
                        )
                    except Exception as e:
                        # Avoid crashing training due to renderer
                        print(f"[Trainer] Renderer.draw() error: {e}")
                        visualize = False  # Disable rendering for the rest of this episode

                # Prepare next step
                obs_left, obs_right = nobs_left, nobs_right
                last_info = info

                # Safety: prevent runaway loop if env misbehaves (shouldn't happen, env enforces max_steps)
                if steps > self.env.max_steps + 5:
                    print("[Trainer] Warning: exceeded expected max_steps; breaking out.")
                    break

                if done:
                    break

        # Close renderer if we opened it for this episode
        if visualize:
            try:
                self.renderer.close()
            except Exception:
                pass

        # Episode result aggregation
        avg_loss_left = (loss_sum_left / loss_count_left) if loss_count_left > 0 else 0.0
        avg_loss_right = (loss_sum_right / loss_count_right) if loss_count_right > 0 else 0.0

        result = {
            "episode": self.episode,
            "steps": steps,
            "total_reward_left": float(total_reward_left),
            "total_reward_right": float(total_reward_right),
            "epsilon_left": float(getattr(self.agent_left, "epsilon", 0.0)),
            "epsilon_right": float(getattr(self.agent_right, "epsilon", 0.0)),
            "avg_loss_left": float(avg_loss_left),
            "avg_loss_right": float(avg_loss_right),
            "goal": last_info.get("goal") if last_info else None,
            "score_left": last_info.get("score_left", getattr(self.env, "score_left", 0)) if last_info else getattr(self.env, "score_left", 0),
            "score_right": last_info.get("score_right", getattr(self.env, "score_right", 0)) if last_info else getattr(self.env, "score_right", 0),
        }
        return result

    def maybe_checkpoint(self) -> None:
        """
        Save a checkpoint every configured number of episodes.
        """
        if self.checkpoint_every <= 0:
            return
        if self.episode % self.checkpoint_every != 0:
            return

        # Path pattern: checkpoint_ep{episode}.pt inside checkpoint_dir
        path = f"{self.checkpoint_dir.rstrip('/')}/checkpoint_ep{self.episode}.pt"
        meta = {
            "episode": self.episode,
            "env_steps": self.total_env_steps,
            "timestamp": time.time(),
            "scores": {
                "left": getattr(self.env, "score_left", 0),
                "right": getattr(self.env, "score_right", 0),
            },
            "trainer": {
                "visualize_every_n": self.visualize_every_n,
                "checkpoint_every": self.checkpoint_every,
                "device": self.device,
            },
        }
        try:
            save_checkpoint(self.agent_left, self.agent_right, path, meta)
            print(f"[Trainer] Saved checkpoint: {path}")
        except Exception as e:
            print(f"[Trainer] Failed to save checkpoint at {path}: {e}")


__all__ = ["Trainer"]