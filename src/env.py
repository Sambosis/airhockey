import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

# The environment is configured via the Config object.
from src.config import Config

# Discrete action mapping: 0=stay, 1=up, 2=down, 3=left, 4=right
# Movement vectors are (dx, dy) in grid units, scaled by mallet.max_speed per step.
ACTIONS: List[Tuple[int, int]] = [
    (0, 0),   # stay
    (0, -1),  # up
    (0, 1),   # down
    (-1, 0),  # left
    (1, 0),   # right
]


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
    left_mallets: List[Mallet]  # Changed from single mallet to list
    right_mallets: List[Mallet]  # Changed from single mallet to list
    step_count: int
    score_left: int
    score_right: int
    goal_height_ratio: float


class AirHockeyEnv:
    """
    Deterministic 2D Air Hockey environment for two-agent self-play.

    - Top-down table with elastic collisions against top/bottom walls.
    - Left/Right edges have goal slots (openings) centered vertically; elsewhere the side walls are solid.
    - Two mallets, discrete actions, constrained to their half of the table.
    - Puck has slight friction per tick.
    - Observations normalized to [-1, 1].
    - Right agent observation is mirrored horizontally to preserve symmetry.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.width = int(config.width)
        self.height = int(config.height)
        self.max_steps = int(config.max_steps)
        self.friction = float(config.friction)
        self.mallet_speed = float(config.mallet_speed)
        self.num_mallets_per_side = int(config.num_mallets_per_side)

        self.puck = Puck(self.width / 2, self.height / 2, 0.0, 0.0, r=config.puck_radius, mass=config.puck_mass)
        
        # Create multiple mallets per side
        self.left_mallets: List[Mallet] = []
        self.right_mallets: List[Mallet] = []
        
        # Position mallets evenly spaced vertically
        for i in range(self.num_mallets_per_side):
            # Calculate vertical position for even spacing
            y_pos = (self.height / (self.num_mallets_per_side + 1)) * (i + 1)
            
            # Left mallets
            self.left_mallets.append(Mallet(
                self.width * 0.25,
                y_pos,
                0.0,
                0.0,
                r=config.mallet_radius,
                side="left",
                max_speed=self.mallet_speed,
            ))
            
            # Right mallets
            self.right_mallets.append(Mallet(
                self.width * 0.75,
                y_pos,
                0.0,
                0.0,
                r=config.mallet_radius,
                side="right",
                max_speed=self.mallet_speed,
            ))

        self.rng: np.random.Generator = np.random.default_rng(config.seed)
        self.step_count: int = 0
        self.score_left: int = 0
        self.score_right: int = 0

        # Normalization constants
        self._puck_vel_norm = config.vel_norm_puck
        self._mallet_vel_norm = config.vel_norm_mallet

        # Goal opening size as a fraction of table height
        self.goal_height_ratio = config.goal_height_ratio

    # Public API

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment to the initial state.

        Returns:
            obs_left: Observation for the left agent.
            obs_right: Mirrored observation for the right agent.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        
        # Reset all mallets to their initial positions
        for i, mallet in enumerate(self.left_mallets):
            y_pos = (self.height / (self.num_mallets_per_side + 1)) * (i + 1)
            mallet.x = self.width * 0.25
            mallet.y = y_pos
            mallet.vx = 0.0
            mallet.vy = 0.0

        for i, mallet in enumerate(self.right_mallets):
            y_pos = (self.height / (self.num_mallets_per_side + 1)) * (i + 1)
            mallet.x = self.width * 0.75
            mallet.y = y_pos
            mallet.vx = 0.0
            mallet.vy = 0.0

        # Puck to center with randomized initial velocity
        self.puck.x = self.width * 0.5
        self.puck.y = self.height * 0.5
        vx, vy = self._random_initial_puck_velocity()
        self.puck.vx = vx
        self.puck.vy = vy

        return self._normalize_obs(mirror_for_right=False), self._normalize_obs(mirror_for_right=True)

    def step(
        self, a_left: int, a_right: int
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, Any]]:
        """
        Advance the simulation by one tick.

        Args:
            a_left: Discrete action for left agent (applies to all left mallets).
            a_right: Discrete action for right agent (applies to all right mallets).

        Returns:
            obs_left, obs_right: Observations for both agents (right is mirrored).
            r_left, r_right: Rewards for both agents.
            done: Whether the episode has terminated.
            info: Dictionary with auxiliary info (scores, step_count, GameState).
        """
        self.step_count += 1

        # Apply mallet actions -> set velocities and positions (constrained)
        # Apply same action to all mallets on each side
        for mallet in self.left_mallets:
            self._apply_mallet_action(mallet, a_left)
        for mallet in self.right_mallets:
            self._apply_mallet_action(mallet, a_right)

        # Physics: move puck, handle wall + mallet collisions, friction
        goal_scored, left_hit, right_hit = self._physics_step()

        # Compute rewards
        r_left, r_right = self._compute_rewards(goal_scored, left_hit, right_hit)

        done = False
        if goal_scored is not None:
            # Update scores and terminate
            if goal_scored == "left":
                self.score_left += 1
            elif goal_scored == "right":
                self.score_right += 1
            done = True
        elif self.step_count >= self.max_steps:
            # Max steps reached; terminal without additional terminal reward
            done = True

        obs_left = self._normalize_obs(mirror_for_right=False)
        obs_right = self._normalize_obs(mirror_for_right=True)

        info = {
            "score_left": self.score_left,
            "score_right": self.score_right,
            "step_count": self.step_count,
            "state": self._snapshot_state(),
            "goal": goal_scored,  # "left"|"right"|None
        }
        return obs_left, obs_right, r_left, r_right, done, info

    # Physics and helpers

    def _apply_mallet_action(self, mallet: Mallet, action: int) -> None:
        """Apply discrete action to mallet with clamping to allowed region; set vx,vy accordingly."""
        if not (0 <= action < len(ACTIONS)):
            raise ValueError(f"Invalid action {action}. Must be in [0,{len(ACTIONS)-1}]")
        move = ACTIONS[action]
        dx = float(move[0]) * mallet.max_speed
        dy = float(move[1]) * mallet.max_speed

        old_x, old_y = mallet.x, mallet.y
        new_x = old_x + dx
        new_y = old_y + dy

        # Clamp within allowed region
        if mallet.side == "left":
            x_min = mallet.r
            x_max = self.width * 0.5 - mallet.r
        else:  # right
            x_min = self.width * 0.5 + mallet.r
            x_max = self.width - mallet.r
        y_min = mallet.r
        y_max = self.height - mallet.r

        # Clamp
        clamped_x = float(np.clip(new_x, x_min, x_max))
        clamped_y = float(np.clip(new_y, y_min, y_max))

        # Set position and resulting velocity from motion
        mallet.x = clamped_x
        mallet.y = clamped_y
        mallet.vx = mallet.x - old_x
        mallet.vy = mallet.y - old_y

    def _physics_step(self) -> Tuple[Optional[str], bool, bool]:
        """
        Advance puck physics one step: integrate, handle collisions, apply friction.

        Returns:
            goal_scored: "left" if left player scored, "right" if right player scored, else None.
            left_hit: True if any left mallet hit the puck.
            right_hit: True if any right mallet hit the puck.
        """
        # Integrate puck
        self.puck.x += self.puck.vx
        self.puck.y += self.puck.vy

        # Handle top/bottom wall collisions (perfectly elastic)
        if self.puck.y - self.puck.r < 0.0:
            self.puck.y = self.puck.r
            self.puck.vy = -self.puck.vy
        elif self.puck.y + self.puck.r > self.height:
            self.puck.y = self.height - self.puck.r
            self.puck.vy = -self.puck.vy

        # Side walls with goal openings centered vertically
        gh = self.height * self.goal_height_ratio
        gy0 = 0.5 * (self.height - gh)
        gy1 = gy0 + gh

        # Left side
        if self.puck.x - self.puck.r < 0.0:
            if gy0 <= self.puck.y <= gy1:
                # In goal opening: allow to pass; count goal if center crosses plane
                if self.puck.x < 0.0:
                    return "right", False, False  # Right agent scores on left goal
            else:
                # Solid wall: reflect
                self.puck.x = self.puck.r
                self.puck.vx = -self.puck.vx

        # Right side
        if self.puck.x + self.puck.r > self.width:
            if gy0 <= self.puck.y <= gy1:
                if self.puck.x > self.width:
                    return "left", False, False  # Left agent scores on right goal
            else:
                self.puck.x = self.width - self.puck.r
                self.puck.vx = -self.puck.vx

        # Mallet collisions - check all mallets
        left_hit = False
        right_hit = False
        
        for mallet in self.left_mallets:
            if self._handle_collisions(mallet):
                left_hit = True
                
        for mallet in self.right_mallets:
            if self._handle_collisions(mallet):
                right_hit = True

        # Apply friction to puck velocity
        self.puck.vx *= self.friction
        self.puck.vy *= self.friction

        return None, left_hit, right_hit

    def _handle_collisions(self, mallet: Mallet) -> bool:
        """Handle elastic collision between mallet (moving) and puck (circle-circle). Returns True if a collision occurred."""
        dx = self.puck.x - mallet.x
        dy = self.puck.y - mallet.y
        dist = math.hypot(dx, dy)
        min_dist = self.puck.r + mallet.r

        if dist >= min_dist:
            return False

        # Normal vector (from mallet to puck)
        if dist > 1e-8:
            nx = dx / dist
            ny = dy / dist
        else:
            # Overlapping centers; choose a deterministic pseudo-random normal
            angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
            nx, ny = math.cos(angle), math.sin(angle)

        # Positional correction to resolve overlap
        overlap = min_dist - dist
        # Move puck out along normal (a small epsilon to avoid re-colliding)
        self.puck.x += nx * (overlap + 1e-3)
        self.puck.y += ny * (overlap + 1e-3)

        # Relative velocity
        rvx = self.puck.vx - mallet.vx
        rvy = self.puck.vy - mallet.vy
        rel_normal = rvx * nx + rvy * ny

        # Reflect relative velocity only if moving towards mallet (rel_normal < 0)
        if rel_normal < 0.0:
            rvx_post = rvx - 2.0 * rel_normal * nx
            rvy_post = rvy - 2.0 * rel_normal * ny
            # New puck velocity is relative + mallet velocity
            self.puck.vx = rvx_post + mallet.vx
            self.puck.vy = rvy_post + mallet.vy
        else:
            # Already separating; keep current velocity after positional fix
            pass
        
        return True

    def _normalize_obs(self, mirror_for_right: bool) -> np.ndarray:
        """
        Build normalized observation vector.
        Layout: [puck_x, puck_y, puck_vx, puck_vy,
                 left_mallet_1_x, left_mallet_1_y, left_mallet_1_vx, left_mallet_1_vy,
                 ...
                 left_mallet_n_x, left_mallet_n_y, left_mallet_n_vx, left_mallet_n_vy,
                 right_mallet_1_x, right_mallet_1_y, right_mallet_1_vx, right_mallet_1_vy,
                 ...
                 right_mallet_n_x, right_mallet_n_y, right_mallet_n_vx, right_mallet_n_vy]

        - For left agent (mirror_for_right=False): observation is as-is.
        - For right agent (mirror_for_right=True): observation is horizontally mirrored and
          mallets are swapped so that right mallets come first.
        """
        def norm_pos_x(x: float) -> float:
            return float(np.clip(2.0 * x / self.width - 1.0, -1.0, 1.0))

        def norm_pos_y(y: float) -> float:
            return float(np.clip(2.0 * y / self.height - 1.0, -1.0, 1.0))

        def norm_v_puck(v: float) -> float:
            return float(np.clip(v / self._puck_vel_norm, -1.0, 1.0))

        def norm_v_mallet(v: float) -> float:
            denom = max(1e-6, self._mallet_vel_norm)
            return float(np.clip(v / denom, -1.0, 1.0))

        obs_list = []
        
        if not mirror_for_right:
            # Puck
            obs_list.extend([
                norm_pos_x(self.puck.x),
                norm_pos_y(self.puck.y),
                norm_v_puck(self.puck.vx),
                norm_v_puck(self.puck.vy)
            ])
            
            # Left mallets
            for mallet in self.left_mallets:
                obs_list.extend([
                    norm_pos_x(mallet.x),
                    norm_pos_y(mallet.y),
                    norm_v_mallet(mallet.vx),
                    norm_v_mallet(mallet.vy)
                ])
            
            # Right mallets
            for mallet in self.right_mallets:
                obs_list.extend([
                    norm_pos_x(mallet.x),
                    norm_pos_y(mallet.y),
                    norm_v_mallet(mallet.vx),
                    norm_v_mallet(mallet.vy)
                ])
        else:
            # Mirror horizontally for right agent
            # Puck (mirrored)
            obs_list.extend([
                norm_pos_x(self.width - self.puck.x),
                norm_pos_y(self.puck.y),
                norm_v_puck(-self.puck.vx),
                norm_v_puck(self.puck.vy)
            ])
            
            # Right mallets first (mirrored to appear as "own" mallets)
            for mallet in self.right_mallets:
                obs_list.extend([
                    norm_pos_x(self.width - mallet.x),
                    norm_pos_y(mallet.y),
                    norm_v_mallet(-mallet.vx),
                    norm_v_mallet(mallet.vy)
                ])
            
            # Left mallets (mirrored to appear as "opponent" mallets)
            for mallet in self.left_mallets:
                obs_list.extend([
                    norm_pos_x(self.width - mallet.x),
                    norm_pos_y(mallet.y),
                    norm_v_mallet(-mallet.vx),
                    norm_v_mallet(mallet.vy)
                ])

        return np.array(obs_list, dtype=np.float32)

    def _compute_rewards(self, goal_scored: Optional[str], left_hit: bool, right_hit: bool) -> Tuple[float, float]:
        """
        Compute per-step rewards for left and right agents including shaping.
        """
        # Hit reward
        hit_reward_left = self.config.reward_on_hit if left_hit else 0.0
        hit_reward_right = self.config.reward_on_hit if right_hit else 0.0

        # Directional component
        dir_left = self.config.reward_toward_opponent if self.puck.vx > 0.0 else 0.0
        dir_right = self.config.reward_toward_opponent if self.puck.vx < 0.0 else 0.0

        # Distance component - use closest mallet distance (normalize by table diagonal)
        diag = math.hypot(self.width, self.height)
        
        # Find closest left mallet to puck
        min_dist_left = float('inf')
        for mallet in self.left_mallets:
            dist = math.hypot(mallet.x - self.puck.x, mallet.y - self.puck.y)
            min_dist_left = min(min_dist_left, dist)
        dl = min_dist_left / diag
        
        # Find closest right mallet to puck
        min_dist_right = float('inf')
        for mallet in self.right_mallets:
            dist = math.hypot(mallet.x - self.puck.x, mallet.y - self.puck.y)
            min_dist_right = min(min_dist_right, dist)
        dr = min_dist_right / diag
        
        dist_left = self.config.reward_distance_weight * (-dl)
        dist_right = self.config.reward_distance_weight * (-dr)

        # Time penalty
        time_pen = self.config.reward_time_penalty

        r_left = dir_left + dist_left + time_pen + hit_reward_left
        r_right = dir_right + dist_right + time_pen + hit_reward_right

        # Terminal goal bonuses
        if goal_scored == "left":
            r_left += self.config.reward_goal
            r_right += self.config.reward_concede
        elif goal_scored == "right":
            r_left += self.config.reward_concede
            r_right += self.config.reward_goal

        return r_left, r_right

    def _random_initial_puck_velocity(self) -> Tuple[float, float]:
        """Randomize initial puck velocity with a reasonable magnitude."""
        # Choose speed in configured range and random direction; ensure |vx| not near zero.
        min_speed, max_speed = self.config.puck_speed_init
        for _ in range(16):
            speed = float(self.rng.uniform(min_speed, max_speed))
            angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            if abs(vx) >= 0.5:  # avoid very small horizontal speed
                return vx, vy
        # Fallback
        return min_speed, 0.0

    def _snapshot_state(self) -> GameState:
        """Create an immutable snapshot GameState for rendering/logging."""
        puck = Puck(self.puck.x, self.puck.y, self.puck.vx, self.puck.vy, r=self.puck.r, mass=self.puck.mass)
        
        left_mallets = []
        for mallet in self.left_mallets:
            left_mallets.append(Mallet(
                mallet.x,
                mallet.y,
                mallet.vx,
                mallet.vy,
                r=mallet.r,
                side=mallet.side,
                max_speed=mallet.max_speed,
            ))
        
        right_mallets = []
        for mallet in self.right_mallets:
            right_mallets.append(Mallet(
                mallet.x,
                mallet.y,
                mallet.vx,
                mallet.vy,
                r=mallet.r,
                side=mallet.side,
                max_speed=mallet.max_speed,
            ))
        
        return GameState(
            puck=puck,
            left_mallets=left_mallets,
            right_mallets=right_mallets,
            step_count=self.step_count,
            score_left=self.score_left,
            score_right=self.score_right,
            goal_height_ratio=self.goal_height_ratio,
        )


__all__ = ["ACTIONS", "AirHockeyEnv", "GameState", "Puck", "Mallet"]