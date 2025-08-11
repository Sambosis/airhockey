from __future__ import annotations

import os
from typing import Optional, Dict, Tuple
from contextlib import suppress

# Ensure pygame does not attempt to initialize audio backends
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from src.env import GameState
from recorder import record_pygame


class Renderer:
    """
    pygame-based renderer for the Air Hockey environment.

    - Does not initialize sound/mixer.
    - Only creates a window/surface when open() is called.
    - draw() is a no-op if the renderer is not open (screen is None).
    - close() releases the display and pygame resources.
    - Cap frame rate to the given FPS during visual episodes.
    """

    def __init__(self, width: int = 800, height: int = 400, fps: int = 60) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font_small: Optional[pygame.font.Font] = None
        self.font_large: Optional[pygame.font.Font] = None

        # Color palette
        self.colors: Dict[str, Tuple[int, int, int]] = {
            "table_bg": (20, 110, 100),
            "center_line": (200, 200, 200),
            "goal_line": (255, 215, 0),
            "puck": (250, 250, 250),
            "left_mallet": (66, 135, 245),   # blue
            "right_mallet": (235, 64, 52),   # red
            "text": (255, 255, 255),
            "shadow": (0, 0, 0),
        }

        # Track whether user requested window close
        self._closed_by_user: bool = False

    @property
    def is_open(self) -> bool:
        return self.screen is not None

    def open(self) -> None:
        """
        Initialize pygame display and fonts; create the rendering window.
        Safe to call multiple times.
        """
        if self.is_open:
            return

        # Initialize only the required pygame subsystems
        if not pygame.display.get_init():
            pygame.display.init()
        if not pygame.font.get_init():
            pygame.font.init()

        try:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Air Hockey Self-Play (DQN)")
        except Exception as e:
            # Fail gracefully: keep renderer disabled (no-op)
            print(f"[Renderer] Failed to initialize display: {e}")
            self.screen = None
            return

        self.clock = pygame.time.Clock()
        # Fonts: Use default fonts for portability
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 28)
        self._closed_by_user = False

    def draw(self, state: GameState, episode_idx: int, eps_left: float, eps_right: float) -> None:
        """
        Draw a single frame based on the provided GameState and HUD info.

        Args:
            state: Snapshot of current game state to render.
            episode_idx: Current episode index (for HUD).
            eps_left: Epsilon value of left agent (for HUD).
            eps_right: Epsilon value of right agent (for HUD).
        """
        if not self.is_open:
            # Non-visual episodes: no-op
            return

        assert self.screen is not None  # for type checkers
        assert self.clock is not None
        assert self.font_small is not None and self.font_large is not None

        # Handle basic window events to keep UI responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed_by_user = True
                self.close()
                return

        # Background (table)
        self.screen.fill(self.colors["table_bg"])

        # Center line
        pygame.draw.line(
            self.screen,
            self.colors["center_line"],
            (self.width // 2, 0),
            (self.width // 2, self.height),
            width=2,
        )

        # Side walls with goal openings centered vertically
        # Read goal opening ratio from state if present; fallback to 0.35
        goal_height_ratio = getattr(state, "goal_height_ratio", 0.35)

        gh = int(self.height * goal_height_ratio)
        gy0 = (self.height - gh) // 2
        gy1 = gy0 + gh

        wall_x_left = 4
        wall_x_right = self.width - 4
        # Left wall: draw two segments (top and bottom), leaving middle open
        pygame.draw.line(self.screen, self.colors["goal_line"], (wall_x_left, 0), (wall_x_left, gy0), width=4)
        pygame.draw.line(self.screen, self.colors["goal_line"], (wall_x_left, gy1), (wall_x_left, self.height), width=4)
        # Right wall
        pygame.draw.line(self.screen, self.colors["goal_line"], (wall_x_right, 0), (wall_x_right, gy0), width=4)
        pygame.draw.line(self.screen, self.colors["goal_line"], (wall_x_right, gy1), (wall_x_right, self.height), width=4)

        # Draw puck with a subtle shadow
        puck = state.puck
        shadow_offset = 2
        pygame.draw.circle(
            self.screen,
            self.colors["shadow"],
            (int(puck.x) + shadow_offset, int(puck.y) + shadow_offset),
            int(puck.r),
        )
        pygame.draw.circle(
            self.screen, self.colors["puck"], (int(puck.x), int(puck.y)), int(puck.r)
        )

        # Draw mallets with shadows
        for m in state.left:
            pygame.draw.circle(
                self.screen,
                self.colors["shadow"],
                (int(m.x) + shadow_offset, int(m.y) + shadow_offset),
                int(m.r),
            )
            pygame.draw.circle(
                self.screen,
                self.colors["left_mallet"],
                (int(m.x), int(m.y)),
                int(m.r),
            )

        for m in state.right:
            pygame.draw.circle(
                self.screen,
                self.colors["shadow"],
                (int(m.x) + shadow_offset, int(m.y) + shadow_offset),
                int(m.r),
            )
            pygame.draw.circle(
                self.screen,
                self.colors["right_mallet"],
                (int(m.x), int(m.y)),
                int(m.r),
            )

        # HUD: Scores, Episode, Epsilons, Step
        hud_margin = 8
        # Scores centered at top
        score_text = f"{state.score_left} : {state.score_right}"
        score_surf = self.font_large.render(score_text, True, self.colors["text"])
        self.screen.blit(
            score_surf,
            (self.width // 2 - score_surf.get_width() // 2, hud_margin),
        )

        # Episode and step at top-left
        info_left = f"Episode: {episode_idx}   Step: {state.step_count}"
        info_left_surf = self.font_small.render(info_left, True, self.colors["text"])
        self.screen.blit(info_left_surf, (hud_margin, hud_margin))

        # Epsilons at top-right
        info_right = f"ε Left: {eps_left:.3f}   ε Right: {eps_right:.3f}"
        info_right_surf = self.font_small.render(info_right, True, self.colors["text"])
        self.screen.blit(
            info_right_surf,
            (self.width - info_right_surf.get_width() - hud_margin, hud_margin),
        )
        # Flip and cap FPS
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        """
        Close the renderer window and quit pygame display modules.
        Safe to call multiple times.
        """
        if self.screen is not None:
            with suppress(Exception):
                pygame.display.quit()
        self.screen = None
        self.clock = None
        self.font_small = None
        self.font_large = None

        # If no other pygame modules are in use, quit pygame entirely
        with suppress(Exception):
            if not pygame.display.get_init():
                pygame.quit()


__all__ = ["Renderer"]
