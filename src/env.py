import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
from numba import njit, prange

# The environment is configured via the Config object.
from src.config import Config

# Discrete action mapping: 0=stay, 1=up, 2=down, 3=left, 4=right
# Movement vectors are (dx, dy) in grid units, scaled by mallet.max_speed per step.
ACTIONS = np.array([
    (0, 0),   # stay
    (0, -1),  # up
    (0, 1),   # down
    (-1, 0),  # left
    (1, 0),   # right
], dtype=np.int64)


@njit
def decode_composite_action(composite_action: int, mallets_per_side: int) -> List[int]:
    """
    Decode a composite action into individual mallet actions.
    
    Args:
        composite_action: Single integer representing actions for all mallets
        mallets_per_side: Number of mallets per side
        
    Returns:
        List of individual actions for each mallet
    """
    actions = []
    remaining = composite_action
    for _ in range(mallets_per_side):
        actions.append(remaining % 5)
        remaining //= 5
    return actions


@dataclass
class Puck:
    x: float
    y: float
    vx: float
    vy: float
    r: float = 10.0
    mass: float = 1.0


@dataclass
class Mallet:
    x: float
    y: float
    vx: float
    vy: float
    r: float = 20.0
    side: str = "left"  # "left" or "right"
    max_speed: float = 6.0


@dataclass
class GameState:
    puck: Puck
    left: List[Mallet]
    right: List[Mallet]
    step_count: int
    score_left: int
    score_right: int
    goal_height_ratio: float


@njit
def _apply_mallet_action_jit(
    mallet_x: float,
    mallet_y: float,
    action: int,
    max_speed: float,
    width: int,
    height: int,
    radius: float,
    side: int,  # 0 for left, 1 for right
) -> Tuple[float, float, float, float]:
    move = ACTIONS[action]
    dx = float(move[0]) * max_speed
    dy = float(move[1]) * max_speed

    old_x, old_y = mallet_x, mallet_y
    new_x = old_x + dx
    new_y = old_y + dy

    if side == 0:  # left
        x_min = radius
        x_max = width * 0.5 - radius
    else:  # right
        x_min = width * 0.5 + radius
        x_max = width - radius
    y_min = radius
    y_max = height - radius

    clamped_x = max(x_min, min(new_x, x_max))
    clamped_y = max(y_min, min(new_y, y_max))

    vx = clamped_x - old_x
    vy = clamped_y - old_y

    return clamped_x, clamped_y, vx, vy


@njit
def _handle_collisions_jit(
    puck_x: float,
    puck_y: float,
    puck_vx: float,
    puck_vy: float,
    puck_r: float,
    mallet_x: float,
    mallet_y: float,
    mallet_vx: float,
    mallet_vy: float,
    mallet_r: float,
) -> Tuple[float, float, float, float, bool]:
    dx = puck_x - mallet_x
    dy = puck_y - mallet_y
    dist = math.hypot(dx, dy)
    min_dist = puck_r + mallet_r

    if dist >= min_dist:
        return puck_x, puck_y, puck_vx, puck_vy, False

    if dist > 1e-8:
        nx = dx / dist
        ny = dy / dist
    else:
        angle = np.random.uniform(0.0, 2.0 * math.pi)
        nx, ny = math.cos(angle), math.sin(angle)

    overlap = min_dist - dist
    puck_x += nx * (overlap + 1e-3)
    puck_y += ny * (overlap + 1e-3)

    rvx = puck_vx - mallet_vx
    rvy = puck_vy - mallet_vy
    rel_normal = rvx * nx + rvy * ny

    if rel_normal < 0.0:
        rvx_post = rvx - 2.0 * rel_normal * nx
        rvy_post = rvy - 2.0 * rel_normal * ny
        puck_vx = rvx_post + mallet_vx
        puck_vy = rvy_post + mallet_vy

    return puck_x, puck_y, puck_vx, puck_vy, True


@njit
def _physics_step_jit(
    puck_x: float,
    puck_y: float,
    puck_vx: float,
    puck_vy: float,
    puck_r: float,
    left_mallets: np.ndarray,
    right_mallets: np.ndarray,
    width: int,
    height: int,
    goal_height_ratio: float,
    friction: float,
) -> Tuple[float, float, float, float, Optional[int], bool, bool]:
    puck_x += puck_vx
    puck_y += puck_vy

    if puck_y - puck_r < 0.0:
        puck_y = puck_r
        puck_vy = -puck_vy
    elif puck_y + puck_r > height:
        puck_y = height - puck_r
        puck_vy = -puck_vy

    gh = height * goal_height_ratio
    gy0 = 0.5 * (height - gh)
    gy1 = gy0 + gh

    goal_scored = -1  # -1 for no goal, 0 for left, 1 for right

    if puck_x - puck_r < 0.0:
        if gy0 <= puck_y <= gy1:
            if puck_x < 0.0:
                goal_scored = 1  # Right agent scores
        else:
            puck_x = puck_r
            puck_vx = -puck_vx

    if puck_x + puck_r > width:
        if gy0 <= puck_y <= gy1:
            if puck_x > width:
                goal_scored = 0  # Left agent scores
        else:
            puck_x = width - puck_r
            puck_vx = -puck_vx

    if goal_scored != -1:
        return puck_x, puck_y, puck_vx, puck_vy, goal_scored, False, False

    left_hit = False
    for i in prange(len(left_mallets)):
        puck_x, puck_y, puck_vx, puck_vy, hit = _handle_collisions_jit(
            puck_x, puck_y, puck_vx, puck_vy, puck_r,
            left_mallets[i, 0], left_mallets[i, 1], left_mallets[i, 2], left_mallets[i, 3], left_mallets[i, 4]
        )
        if hit:
            left_hit = True

    right_hit = False
    for i in prange(len(right_mallets)):
        puck_x, puck_y, puck_vx, puck_vy, hit = _handle_collisions_jit(
            puck_x, puck_y, puck_vx, puck_vy, puck_r,
            right_mallets[i, 0], right_mallets[i, 1], right_mallets[i, 2], right_mallets[i, 3], right_mallets[i, 4]
        )
        if hit:
            right_hit = True

    puck_vx *= friction
    puck_vy *= friction

    return puck_x, puck_y, puck_vx, puck_vy, None, left_hit, right_hit


class AirHockeyEnv:
    """
    Deterministic 2D Air Hockey environment for two-agent self-play.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.width = int(config.width)
        self.height = int(config.height)
        self.max_steps = int(config.max_steps)
        self.friction = float(config.friction)
        self.mallet_speed = float(config.mallet_speed)

        self.puck = Puck(self.width / 2, self.height / 2, 0.0, 0.0, r=config.puck_radius, mass=config.puck_mass)
        
        self.mallets_per_side = config.mallets_per_side
        self.left_mallets = []
        self.right_mallets = []
        
        for i in range(self.mallets_per_side):
            y_pos = self.height * (i + 1) / (self.mallets_per_side + 1)
            
            left_mallet = Mallet(
                self.width * 0.25, y_pos, 0.0, 0.0,
                r=config.mallet_radius, side="left", max_speed=self.mallet_speed,
            )
            self.left_mallets.append(left_mallet)
            
            right_mallet = Mallet(
                self.width * 0.75, y_pos, 0.0, 0.0,
                r=config.mallet_radius, side="right", max_speed=self.mallet_speed,
            )
            self.right_mallets.append(right_mallet)
        
        self.left_mallet = self.left_mallets[0] if self.left_mallets else None
        self.right_mallet = self.right_mallets[0] if self.right_mallets else None

        self.rng: np.random.Generator = np.random.default_rng(config.seed)
        self.step_count: int = 0
        self.score_left: int = 0
        self.score_right: int = 0

        self._puck_vel_norm = config.vel_norm_puck
        self._mallet_vel_norm = config.vel_norm_mallet
        self.goal_height_ratio = config.goal_height_ratio

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        for i, mallet in enumerate(self.left_mallets):
            y_pos = self.height * (i + 1) / (self.mallets_per_side + 1)
            mallet.x = self.width * 0.25
            mallet.y = y_pos
            mallet.vx = 0.0
            mallet.vy = 0.0

        for i, mallet in enumerate(self.right_mallets):
            y_pos = self.height * (i + 1) / (self.mallets_per_side + 1)
            mallet.x = self.width * 0.75
            mallet.y = y_pos
            mallet.vx = 0.0
            mallet.vy = 0.0

        self.puck.x = self.width * 0.5
        self.puck.y = self.height * 0.5
        vx, vy = self._random_initial_puck_velocity()
        self.puck.vx = vx
        self.puck.vy = vy

        return self._normalize_obs(mirror_for_right=False), self._normalize_obs(mirror_for_right=True)

    def step(
        self, a_left: int, a_right: int
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, Any]]:
        self.step_count += 1

        max_action = 5 ** self.mallets_per_side - 1
        if not (0 <= a_left <= max_action):
            raise ValueError(f"Invalid left action {a_left}. Must be in [0, {max_action}]")
        if not (0 <= a_right <= max_action):
            raise ValueError(f"Invalid right action {a_right}. Must be in [0, {max_action}]")

        left_actions = decode_composite_action(a_left, self.mallets_per_side)
        right_actions = decode_composite_action(a_right, self.mallets_per_side)

        for i, action in enumerate(left_actions):
            self._apply_mallet_action(self.left_mallets[i], action)
        for i, action in enumerate(right_actions):
            self._apply_mallet_action(self.right_mallets[i], action)

        left_mallets_np = np.array([[m.x, m.y, m.vx, m.vy, m.r] for m in self.left_mallets])
        right_mallets_np = np.array([[m.x, m.y, m.vx, m.vy, m.r] for m in self.right_mallets])

        puck_x, puck_y, puck_vx, puck_vy, goal_scored_code, left_hit, right_hit = _physics_step_jit(
            self.puck.x, self.puck.y, self.puck.vx, self.puck.vy, self.puck.r,
            left_mallets_np, right_mallets_np,
            self.width, self.height, self.goal_height_ratio, self.friction
        )

        self.puck.x, self.puck.y, self.puck.vx, self.puck.vy = puck_x, puck_y, puck_vx, puck_vy
        
        goal_scored = None
        if goal_scored_code is not None:
            if goal_scored_code == 0:
                goal_scored = "left"
            elif goal_scored_code == 1:
                goal_scored = "right"


        r_left, r_right = self._compute_rewards(goal_scored, left_hit, right_hit)

        done = False
        if goal_scored is not None:
            if goal_scored == "left":
                self.score_left += 1
            elif goal_scored == "right":
                self.score_right += 1
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        obs_left = self._normalize_obs(mirror_for_right=False)
        obs_right = self._normalize_obs(mirror_for_right=True)

        info = {
            "score_left": self.score_left,
            "score_right": self.score_right,
            "step_count": self.step_count,
            "state": self._snapshot_state(),
            "goal": goal_scored,
        }
        return obs_left, obs_right, r_left, r_right, done, info

    def _apply_mallet_action(self, mallet: Mallet, action: int) -> None:
        side_code = 0 if mallet.side == "left" else 1
        mallet.x, mallet.y, mallet.vx, mallet.vy = _apply_mallet_action_jit(
            mallet.x, mallet.y, action, mallet.max_speed, self.width, self.height, mallet.r, side_code
        )

    def _normalize_obs(self, mirror_for_right: bool) -> np.ndarray:
        if not mirror_for_right:
            puck = (self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)
            own_mallets = [(m.x, m.y, m.vx, m.vy) for m in self.left_mallets]
            opp_mallets = [(m.x, m.y, m.vx, m.vy) for m in self.right_mallets]
        else:
            puck = (self.width - self.puck.x, self.puck.y, -self.puck.vx, self.puck.vy)
            own_mallets = [
                (self.width - m.x, m.y, -m.vx, m.vy) for m in self.right_mallets
            ]
            opp_mallets = [
                (self.width - m.x, m.y, -m.vx, m.vy) for m in self.left_mallets
            ]

        def norm_pos_x(x: float) -> float:
            return float(np.clip(2.0 * x / self.width - 1.0, -1.0, 1.0))

        def norm_pos_y(y: float) -> float:
            return float(np.clip(2.0 * y / self.height - 1.0, -1.0, 1.0))

        def norm_v_puck(v: float) -> float:
            return float(np.clip(v / self._puck_vel_norm, -1.0, 1.0))

        def norm_v_mallet(v: float) -> float:
            denom = max(1e-6, self._mallet_vel_norm)
            return float(np.clip(v / denom, -1.0, 1.0))

        obs_values = [
            norm_pos_x(puck[0]),
            norm_pos_y(puck[1]),
            norm_v_puck(puck[2]),
            norm_v_puck(puck[3]),
        ]
        
        for mallet in own_mallets:
            obs_values.extend([
                norm_pos_x(mallet[0]),
                norm_pos_y(mallet[1]),
                norm_v_mallet(mallet[2]),
                norm_v_mallet(mallet[3]),
            ])
        
        for mallet in opp_mallets:
            obs_values.extend([
                norm_pos_x(mallet[0]),
                norm_pos_y(mallet[1]),
                norm_v_mallet(mallet[2]),
                norm_v_mallet(mallet[3]),
            ])
        
        obs = np.array(obs_values, dtype=np.float32)
        return obs

    def _compute_rewards(self, goal_scored: Optional[str], left_hit: bool, right_hit: bool) -> Tuple[float, float]:
        hit_reward_left = self.config.reward_on_hit if left_hit else 0.0
        hit_reward_right = self.config.reward_on_hit if right_hit else 0.0

        dir_left = self.config.reward_toward_opponent if self.puck.vx > 0.0 else 0.0
        dir_right = self.config.reward_toward_opponent if self.puck.vx < 0.0 else 0.0

        diag = math.hypot(self.width, self.height)
        dl = min(math.hypot(m.x - self.puck.x, m.y - self.puck.y) for m in self.left_mallets) / diag
        dr = min(math.hypot(m.x - self.puck.x, m.y - self.puck.y) for m in self.right_mallets) / diag
        dist_left = self.config.reward_distance_weight * (-dl)
        dist_right = self.config.reward_distance_weight * (-dr)

        time_pen = self.config.reward_time_penalty

        r_left = dir_left + dist_left + time_pen + hit_reward_left
        r_right = dir_right + dist_right + time_pen + hit_reward_right

        if goal_scored == "left":
            r_left += self.config.reward_goal
            r_right += self.config.reward_concede
        elif goal_scored == "right":
            r_left += self.config.reward_concede
            r_right += self.config.reward_goal

        return r_left, r_right

    def _random_initial_puck_velocity(self) -> Tuple[float, float]:
        min_speed, max_speed = self.config.puck_speed_init
        for _ in range(16):
            speed = float(self.rng.uniform(min_speed, max_speed))
            angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            if abs(vx) >= 0.5:
                return vx, vy
        return min_speed, 0.0

    def _snapshot_state(self) -> GameState:
        puck = Puck(self.puck.x, self.puck.y, self.puck.vx, self.puck.vy, r=self.puck.r, mass=self.puck.mass)
        
        left_mallets = []
        for mallet in self.left_mallets:
            left_mallets.append(Mallet(
                mallet.x, mallet.y, mallet.vx, mallet.vy,
                r=mallet.r, side=mallet.side, max_speed=mallet.max_speed,
            ))
        
        right_mallets = []
        for mallet in self.right_mallets:
            right_mallets.append(Mallet(
                mallet.x, mallet.y, mallet.vx, mallet.vy,
                r=mallet.r, side=mallet.side, max_speed=mallet.max_speed,
            ))
        
        return GameState(
            puck=puck,
            left=left_mallets,
            right=right_mallets,
            step_count=self.step_count,
            score_left=self.score_left,
            score_right=self.score_right,
            goal_height_ratio=self.goal_height_ratio,
        )


__all__ = ["ACTIONS", "AirHockeyEnv", "GameState", "Puck", "Mallet"]
