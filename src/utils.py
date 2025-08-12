from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch

# Internal imports for typing only to avoid circular dependencies at runtime
if TYPE_CHECKING:
    from src.env import AirHockeyEnv  # noqa: F401
    from src.agents.dqn import DQNAgent  # noqa: F401
    from src.agents.dueling_dqn import DuelingDQNAgent  # noqa: F401
    from src.config import Config  # noqa: F401


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """
    Ensure observation is float32 and clipped to [-1, 1].

    Env already provides normalized observations. This function safeguards
    dtype and bounds.

    Args:
        obs: Observation array (any dtype/shape compatible with 1D obs).

    Returns:
        Clipped float32 numpy array with same shape as input.
    """
    arr = np.asarray(obs, dtype=np.float32)
    return np.clip(arr, -1.0, 1.0, out=arr)


def epsilon_by_frame(frame_idx: int, eps_start: float, eps_end: float, decay_frames: int) -> float:
    """
    Linear epsilon schedule from eps_start to eps_end over decay_frames.

    Args:
        frame_idx: Current global frame index (>= 0).
        eps_start: Initial epsilon at frame 0.
        eps_end: Final epsilon after decay_frames (floor).
        decay_frames: Number of frames to decay over (must be > 0).

    Returns:
        Epsilon value for the given frame, clamped to [eps_end, eps_start].
    """
    if decay_frames <= 0:
        return float(eps_end if frame_idx > 0 else eps_start)
    # Linear interpolation
    t = min(max(frame_idx, 0), decay_frames)
    eps = eps_start + (eps_end - eps_start) * (t / float(decay_frames))
    # Clamp
    if eps_start >= eps_end:
        eps = float(np.clip(eps, eps_end, eps_start))
    else:
        eps = float(np.clip(eps, eps_start, eps_end))
    return eps


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch for deterministic behavior (as much as possible).

    Also sets PYTHONHASHSEED and configures cuDNN to be deterministic where supported.

    Args:
        seed: Integer seed value.
    """
    s = int(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)

    if torch is not None:
        torch.manual_seed(s)
        try:
            if torch.cuda.is_available():  # pragma: no cover - environment-dependent
                torch.cuda.manual_seed_all(s)
            # Make cuDNN deterministic if available (may impact performance)
            if hasattr(torch.backends, "cudnn"):  # pragma: no cover - environment-dependent
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            # Be resilient if certain backends are not available
            pass


def compute_shaping(env: "AirHockeyEnv", for_left: bool) -> float:
    """
    Compute per-step shaping reward according to specification from the agent's perspective.

    This mirrors the shaping used internally by the environment:
      - +0.01 if puck x-velocity is toward the opponent
        (left agent favors vx > 0; right agent favors vx < 0)
      - +0.001 scaled by negative distance between the agent mallet and puck
        (distance normalized by table diagonal)
      - -0.001 time penalty per step

    Note: The environment already applies shaping; this helper is provided for logging/consistency.

    Args:
        env: AirHockeyEnv instance.
        for_left: True for left agent's perspective; False for right agent.

    Returns:
        Shaping reward (float).
    """
    # Direction component
    dir_term = 0.01 if (env.puck.vx > 0.0 if for_left else env.puck.vx < 0.0) else 0.0

    # Distance component (normalize by table diagonal)
    diag = float(np.hypot(env.width, env.height))
    if for_left:
        dx = env.left_mallet.x - env.puck.x
        dy = env.left_mallet.y - env.puck.y
    else:
        dx = env.right_mallet.x - env.puck.x
        dy = env.right_mallet.y - env.puck.y
    dist_norm = float(np.hypot(dx, dy)) / (diag if diag > 0.0 else 1.0)
    dist_term = 0.001 * (-dist_norm)

    # Time penalty
    time_penalty = -0.001

    return float(dir_term + dist_term + time_penalty)


def save_checkpoint(
    agent_left: "DQNAgent",
    agent_right: "DQNAgent",
    path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save both agents' states (networks, optimizers, and metadata) to a single checkpoint file.

    Args:
        agent_left: Left agent instance.
        agent_right: Right agent instance.
        path: Destination file path (e.g., 'checkpoints/airhockey_ckpt.pt').
        meta: Optional additional metadata to include (e.g., episode count, total steps).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _agent_payload(agent: Any) -> Dict[str, Any]:
        # Build a resilient payload without assuming exact class shape
        payload: Dict[str, Any] = {}
        # Networks
        q_net = getattr(agent, "q_net", None)
        target_net = getattr(agent, "target_net", None)
        optimizer = getattr(agent, "optimizer", None)

        if q_net is not None:
            payload["q_net"] = q_net.state_dict()
        if target_net is not None:
            payload["target_net"] = target_net.state_dict()
        if optimizer is not None:
            try:
                payload["optimizer"] = optimizer.state_dict()
            except Exception:
                payload["optimizer"] = None

        # Metadata
        agent_meta: Dict[str, Any] = {
            "epsilon": float(getattr(agent, "epsilon", 0.0)),
            "epsilon_min": float(getattr(agent, "epsilon_min", 0.0)),
            "eps_start": float(getattr(agent, "eps_start", getattr(agent, "epsilon", 0.0))),
            "epsilon_decay_steps": int(getattr(agent, "epsilon_decay_steps", 0)),
            "gamma": float(getattr(agent, "gamma", 0.99)),
            "obs_dim": int(getattr(agent, "obs_dim", 0)),
            "action_space_n": int(getattr(agent, "action_space_n", 0)),
            "batch_size": int(getattr(agent, "batch_size", 0)),
            "learn_start": int(getattr(agent, "learn_start", 0)),
            "target_sync": int(getattr(agent, "target_sync", 0)),
            "frame_idx": int(getattr(agent, "frame_idx", 0)),
            "gradient_steps": int(getattr(agent, "gradient_steps", 0)),
            "device": str(getattr(agent, "device", "cpu")),
        }
        payload["meta"] = agent_meta
        return payload

    checkpoint: Dict[str, Any] = {
        "left": _agent_payload(agent_left),
        "right": _agent_payload(agent_right),
        "meta": dict(meta or {}),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    agent_left: "DQNAgent",
    agent_right: "DQNAgent",
    path: str,
    map_location: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """
    Load both agents' states (networks, optimizers, and metadata) from a checkpoint file.

    Args:
        agent_left: Left agent instance to restore into.
        agent_right: Right agent instance to restore into.
        path: Source file path.
        map_location: Optional torch device mapping. Defaults to agent_left.device if available.

    Returns:
        The top-level metadata dictionary saved alongside the agents.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint does not contain expected keys.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if map_location is None:
        map_location = getattr(agent_left, "device", "cpu")

    checkpoint: Dict[str, Any] = torch.load(path, map_location=map_location)

    # Backward compatibility: support single-agent payloads as well
    if "left" not in checkpoint or "right" not in checkpoint:
        # Try to interpret as a single agent checkpoint and load into both for convenience
        single = checkpoint
        for agent in (agent_left, agent_right):
            _restore_agent_from_payload(agent, single)
        return checkpoint.get("meta", {})

    _restore_agent_from_payload(agent_left, checkpoint["left"])
    _restore_agent_from_payload(agent_right, checkpoint["right"])

    return checkpoint.get("meta", {})


def _restore_agent_from_payload(agent: Any, payload: Dict[str, Any]) -> None:
    """
    Internal helper to restore a single agent from a payload dictionary.
    """
    q_net = getattr(agent, "q_net", None)
    target_net = getattr(agent, "target_net", None)
    optimizer = getattr(agent, "optimizer", None)

    if q_net is not None and "q_net" in payload:
        q_net.load_state_dict(payload["q_net"])
    if target_net is not None and "target_net" in payload:
        target_net.load_state_dict(payload["target_net"])
    if optimizer is not None and payload.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception:
            # Optimizer states can be device-specific; ignore failures gracefully
            pass

    meta: Dict[str, Any] = payload.get("meta", {})
    # Restore known scalar attributes if present
    for attr, cast in (
        ("epsilon", float),
        ("epsilon_min", float),
        ("eps_start", float),
        ("epsilon_decay_steps", int),
        ("gamma", float),
        ("batch_size", int),
        ("learn_start", int),
        ("target_sync", int),
        ("frame_idx", int),
        ("gradient_steps", int),
    ):
        if attr in meta:
            try:
                setattr(agent, attr, cast(meta[attr]))
            except Exception:
                pass
    # obs_dim and action_space_n are not modified to avoid architectural mismatch


def update_moving_average(prev: Optional[float], new_value: float, alpha: float = 0.1) -> float:
    """
    Exponential moving average helper for logging.

    Args:
        prev: Previous EMA value (or None to initialize).
        new_value: New sample value.
        alpha: Smoothing factor in (0,1], higher = more responsive.

    Returns:
        Updated EMA value.
    """
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    if prev is None or not np.isfinite(prev):
        return float(new_value)
    return float((1.0 - alpha) * prev + alpha * new_value)


__all__ = [
    "preprocess_obs",
    "epsilon_by_frame",
    "seed_everything",
    "compute_shaping",
    "save_checkpoint",
    "load_checkpoint",
    "update_moving_average",
]