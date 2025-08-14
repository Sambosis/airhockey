from __future__ import annotations
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

# Rich pretty logging (optional)
try:  # pragma: no cover
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except Exception:  # pragma: no cover
    Console = None  # type: ignore
    Table = None  # type: ignore
    _RICH = False

from src.env import AirHockeyEnv, GameState
from src.render import Renderer
from typing import Protocol

class _AgentAPI(Protocol):
    epsilon: float
    def select_action(self, obs, training: bool = True) -> int: ...
    def remember(self, s, a, r, s2, done) -> None: ...
    def update(self) -> Dict[str, Any]: ...


from src.utils import save_checkpoint
from recorder import record_pygame


class Trainer:
    """Self-play trainer with Rich-formatted episode summaries."""

    def __init__(
        self,
        env: AirHockeyEnv,
        agent_left: _AgentAPI,
        agent_right: _AgentAPI,
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
        self.episode = 0
        self.total_env_steps = 0
        self.recent_returns_left: deque[float] = deque(maxlen=100)
        self.recent_returns_right: deque[float] = deque(maxlen=100)
        self.recent_lengths: deque[int] = deque(maxlen=100)
        self.rng = np.random.default_rng(seed)
        self.console: Any = None
        if _RICH and Console is not None:
            try:
                self.console = Console(width=140)
            except Exception:
                self.console = None

    # ------------------------------------------------------------------
    def _log_episode(self, ep_index: int, result: Dict[str, Any], ep_dur: float, visualize: bool) -> None:
        avg_len = float(np.mean(self.recent_lengths)) if self.recent_lengths else result["steps"]
        avg_ret_l = float(np.mean(self.recent_returns_left)) if self.recent_returns_left else result["total_reward_left"]
        avg_ret_r = float(np.mean(self.recent_returns_right)) if self.recent_returns_right else result["total_reward_right"]
        steps_per_sec = result["steps"] / ep_dur if ep_dur > 0 else 0.0
        if _RICH and self.console is not None and Table is not None:
            table = Table(show_header=True, header_style="bold cyan", expand=False, padding=(0, 1))
            for col, just in (
                ("Ep", "right"), ("Steps", "right"), ("Goal", "center"),
                ("R_left", "right"), ("R_right", "right"), ("Eps L/R", "center"),
                ("AvgLen", "right"), ("AvgR L/R", "center"), ("Steps/s", "right"), ("Flags", "left")):
                table.add_column(col, justify=just)
            goal = result.get("goal") or "—"
            def fmt(r: float) -> str:
                return f"[green]{r:+.3f}[/]" if r >= 0 else f"[red]{r:+.3f}[/]"
            table.add_row(
                str(ep_index), str(result["steps"]), str(goal),
                fmt(result["total_reward_left"]), fmt(result["total_reward_right"]),
                f"{result['epsilon_left']:.3f}/{result['epsilon_right']:.3f}",
                f"{avg_len:.1f}", f"{avg_ret_l:+.3f}/{avg_ret_r:+.3f}", f"{steps_per_sec:.2f}",
                "viz" if visualize else "")
            self.console.print(table)
        else:
            print(
                f"[Episode {ep_index:6d}] steps={result['steps']:4d} "
                f"goal={result['goal'] or 'none':>5s} "
                f"R_left={result['total_reward_left']:+.3f} "
                f"R_right={result['total_reward_right']:+.3f} "
                f"eps(L/R)=({result['epsilon_left']:.3f}/{result['epsilon_right']:.3f}) "
                f"avg_len={avg_len:.1f} "
                f"avg_R(L/R)=({avg_ret_l:+.3f}/{avg_ret_r:+.3f}) "
                f"{('(viz) ' if visualize else '')}steps/s={steps_per_sec:.2f}")

    # ------------------------------------------------------------------
    def run(self, num_episodes: int) -> None:
        num_episodes = int(num_episodes)
        if num_episodes <= 0:
            raise ValueError("num_episodes must be > 0")
        start = time.time()
        for _ in range(num_episodes):
            self.episode += 1
            ep = self.episode
            visualize = self.visualize_every_n > 0 and (ep % self.visualize_every_n == 0)
            t0 = time.time()
            result = self.run_episode(visualize=visualize)
            dur = time.time() - t0
            self.recent_returns_left.append(result["total_reward_left"])
            self.recent_returns_right.append(result["total_reward_right"])
            self.recent_lengths.append(result["steps"])
            self._log_episode(ep, result, dur, visualize)
            try:
                self.maybe_checkpoint()
            except Exception as e:  # pragma: no cover
                print(f"[Trainer] Warning: checkpoint failed episode {self.episode}: {e}")
        total_dur = time.time() - start
        msg = (
            f"Finished {num_episodes} episodes in {total_dur/60.0:.2f} min. "
            f"Total env steps: {self.total_env_steps}"
        )
        if _RICH and self.console is not None:
            self.console.rule("Training Complete")
            self.console.print(msg, style="bold green")
        else:
            print(f"[Trainer] {msg}")

    # ------------------------------------------------------------------
    def run_episode(self, visualize: bool) -> Dict[str, Any]:
        if visualize:
            try:
                self.renderer.open()
            except Exception as e:
                print(f"[Trainer] Renderer.open() failed: {e}")
                visualize = False
        obs_left, obs_right = self.env.reset()
        done = False
        steps = 0
        total_reward_left = 0.0
        total_reward_right = 0.0
        loss_sum_left = loss_sum_right = 0.0
        loss_count_left = loss_count_right = 0
        last_info: Dict[str, Any] = {}
        with record_pygame(f"videos/replay_hockey_{self.episode}.mp4", fps=60):
            while not done:
                a_left = self.agent_left.select_action(obs_left, training=True)
                a_right = self.agent_right.select_action(obs_right, training=True)
                nobs_left, nobs_right, r_left, r_right, done, info = self.env.step(a_left, a_right)
                steps += 1
                self.total_env_steps += 1
                self.agent_left.remember(obs_left, a_left, r_left, nobs_left, done)
                self.agent_right.remember(obs_right, a_right, r_right, nobs_right, done)
                m_left = self.agent_left.update()
                m_right = self.agent_right.update()
                if m_left and "loss" in m_left:
                    loss_sum_left += float(m_left["loss"])
                    loss_count_left += 1
                if m_right and "loss" in m_right:
                    loss_sum_right += float(m_right["loss"])
                    loss_count_right += 1
                total_reward_left += r_left
                total_reward_right += r_right
                if visualize and "state" in info and isinstance(info["state"], GameState):
                    try:
                        self.renderer.draw(
                            state=info["state"],
                            episode_idx=self.episode,
                            eps_left=float(getattr(self.agent_left, "epsilon", 0.0)),
                            eps_right=float(getattr(self.agent_right, "epsilon", 0.0)),
                        )
                    except Exception as e:
                        print(f"[Trainer] Renderer.draw() error: {e}")
                        visualize = False
                obs_left, obs_right = nobs_left, nobs_right
                last_info = info
                if steps > self.env.max_steps + 5:
                    print("[Trainer] Warning: exceeded expected max_steps; breaking out.")
                    break
        if visualize:
            try:
                self.renderer.close()
            except Exception:
                pass
        avg_loss_left = (loss_sum_left / loss_count_left) if loss_count_left else 0.0
        avg_loss_right = (loss_sum_right / loss_count_right) if loss_count_right else 0.0
        return {
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

    # ------------------------------------------------------------------
    def maybe_checkpoint(self) -> None:
        if self.checkpoint_every <= 0 or self.episode % self.checkpoint_every != 0:
            return
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
        except Exception as e:  # pragma: no cover
            print(f"[Trainer] Failed to save checkpoint at {path}: {e}")


__all__ = ["Trainer"]