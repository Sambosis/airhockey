from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any
import os

# PyTorch is required by the project; used here to resolve device selection
try:
    import torch
except Exception as e:  # pragma: no cover - torch is a project dependency
    torch = None  # type: ignore


def resolve_device(device_pref: str) -> str:
    """
    Resolve a device preference string into a concrete device identifier.

    Rules:
    - "cpu" -> "cpu"
    - "cuda" -> "cuda" if available else falls back to "cpu"
    - "mps" -> "mps" if available else falls back to "cpu"
    - "auto" -> prefer "cuda" if available, then "mps", else "cpu"

    Returns a string suitable for torch.device(...).
    """
    if torch is None:
        return "cpu"

    pref = device_pref.strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        # Apple Metal Performance Shaders (for Apple Silicon)
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]
        return "mps" if has_mps else "cpu"

    # Auto selection
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]
        if has_mps:
            return "mps"
        return "cpu"

    # Unknown preference, fall back to CPU
    return "cpu"


@dataclass
class Config:
    """
    Centralized configuration for the self-play Air Hockey project.

    This dataclass holds environment dimensions, physics parameters,
    RL hyperparameters, rendering cadence, and checkpointing options.
    It also exposes a helper to resolve 'device'='auto' to an actual
    torch device string.

    Note: Units
    - Positions, radii: pixels
    - Velocities: pixels per simulation tick (dt)
    - dt: seconds per physics step
    """

    # Environment geometry
    width: int = 800
    height: int = 400

    # Simulation/physics
    dt: float = 1.0 / 120.0
    friction: float = 0.997  # multiplicative velocity decay each step

    # Objects
    puck_radius: float = 10.0
    puck_mass: float = 1.0
    mallet_radius: float = 20.0
    mallet_speed: float = 9.0  # max step speed per tick (px)
    mallets_per_side: int = 2  # number of mallets each side controls

    # Puck initial speed (randomized magnitude range, px per tick)
    puck_speed_init: Tuple[float, float] = (3.0, 6.0)

    # Episode control
    max_steps: int = 2000

    # Goal opening (as a fraction of table height, 0<ratio<1)
    goal_height_ratio: float = 0.20

    # Rewards
    reward_goal: float = 50.0
    reward_concede: float = -30.0
    reward_time_penalty: float = -0.01
    reward_toward_opponent: float = 0.0002
    reward_distance_weight: float = 0.0002  # applied to negative distance
    reward_on_hit: float = 5.0  # reward for hitting the puck

    # Observation normalization
    # Positions typically scaled to [-1, 1] using table size directly.
    # Velocity normalization constants used by env/utils during scaling.
    vel_norm_puck: float = 5.0  # expected max magnitude for puck velocity (px/tick)
    vel_norm_mallet: float = 9.0  # equals mallet_speed by default
    mirror_right_obs: bool = True  # mirror horizontally for right agent's perspective

    # Algorithm selection: "dqn" or "dueling"
    algo: str = "dueling"  # Changed from "dqn" to "dueling" for better performance

    # DQN / RL hyperparameters
    lr: float = 5e-4  # Increased from 1e-4 for faster initial learning
    gamma: float = 0.995  # Increased from 0.99 for better long-term planning
    batch_size: int = 256  # Reduced from 1024 for more frequent updates
    buffer_capacity: int = 500_000  # Reduced from 1_000_000 for memory efficiency
    learn_start: int = 1_000  # Reduced from 5_000 for faster learning start
    target_sync: int = 10_000  # Reduced from 50_000 for more frequent target updates

    # Epsilon-greedy exploration
    eps_start: float = 1.0
    eps_end: float = 0.01  # Increased from 0.005 for more exploration
    eps_decay_frames: int = 1_000_000  # Reduced from 3_000_000 for faster decay

    # Training control
    episodes: int = 10_000
    seed: int = 42

    # Rendering
    render_fps: int = 60
    visualize_every_n: int = 50  # Reduced from 100 for more frequent visualization
    no_sound: bool = True  # ensure pygame mixer is not initialized

    # Checkpointing
    checkpoint_every: int = 50  # Reduced from 100 for more frequent checkpoints
    checkpoint_dir: str = "checkpoints"
    load_latest: bool = True  # load latest checkpoint on startup if available

    # Device selection: "auto" | "cpu" | "cuda" | "mps"
    device: str = "auto"

    # Derived/advanced options (rarely changed)
    action_space_n: int = 5
    obs_dim: int = 4  # will be recalculated in __post_init__ based on mallets_per_side

    # Internal: resolved device string (computed in __post_init__)
    _resolved_device: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate basic ranges
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers.")
        if not (0.0 < self.friction <= 1.0):
            raise ValueError("friction must be in (0, 1].")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be > 0.")
        if self.learn_start < 0:
            raise ValueError("learn_start must be >= 0.")
        if self.target_sync <= 0:
            raise ValueError("target_sync must be > 0.")
        if self.episodes <= 0:
            raise ValueError("episodes must be > 0.")
        if self.eps_start < 0.0 or self.eps_end < 0.0 or self.eps_start < self.eps_end:
            raise ValueError("Require eps_start >= eps_end >= 0.0")
        if self.eps_decay_frames <= 0:
            raise ValueError("eps_decay_frames must be > 0.")
        if self.puck_speed_init[0] < 0.0 or self.puck_speed_init[1] <= 0.0 or self.puck_speed_init[1] < self.puck_speed_init[0]:
            raise ValueError("puck_speed_init must be a valid (min, max) with 0 <= min <= max.")
        if not (0.05 <= self.goal_height_ratio <= 0.8):
            raise ValueError("goal_height_ratio must be in [0.05, 0.8] for reasonable gameplay.")

        # Validate mallets per side
        if not (1 <= self.mallets_per_side <= 4):
            raise ValueError("mallets_per_side must be between 1 and 4 for reasonable gameplay.")

        # Calculate observation dimension: puck (4) + mallets_per_side * 4 * 2 (both sides)
        self.obs_dim = 4 + (self.mallets_per_side * 4 * 2)

        # Calculate action space: 5 actions per mallet
        self.action_space_n = 5 ** self.mallets_per_side

        # Keep velocity normalization consistent with configured limits
        self.vel_norm_mallet = float(self.mallet_speed)
        # If user provided a smaller vel_norm_puck than mallet speed, bump to at least mallet speed
        if self.vel_norm_puck < self.mallet_speed:
            self.vel_norm_puck = float(self.mallet_speed)

        # Resolve device preference now for stable downstream use
        self._resolved_device = resolve_device(self.device)

        # Make sure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # If requested, mute audio globally for pygame by setting SDL_AUDIODRIVER=dummy
        if self.no_sound:
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    @property
    def torch_device(self) -> "torch.device":
        """
        Returns a torch.device corresponding to the resolved device preference.
        Falls back to CPU device if torch is unavailable.
        """
        if torch is None:  # pragma: no cover
            class _DummyDevice:
                type = "cpu"

                def __repr__(self) -> str:
                    return "device(type='cpu')"

            return _DummyDevice()  # type: ignore[return-value]
        return torch.device(self._resolved_device)

    @property
    def device_str(self) -> str:
        """
        Returns the resolved device string ("cpu", "cuda", or "mps").
        """
        return self._resolved_device

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the configuration to a dictionary, adding resolved device info.
        """
        d = asdict(self)
        d["device_resolved"] = self._resolved_device
        return d
