import os
import sys
import glob
import traceback
import argparse
from contextlib import suppress
from typing import Optional, Any, Dict, Tuple
from dataclasses import fields as dataclass_fields

# External libs (used minimally here)
try:
    import torch
except Exception:  # pragma: no cover - torch should be available in the environment
    torch = None  # Fallback; device will default to CPU

# Internal project imports
from src.config import Config
from src.env import AirHockeyEnv
from src.agents.dqn import DQNAgent
from src.render import Renderer
from src.train import Trainer
from src.utils import seed_everything, load_checkpoint

# Optional import to satisfy linter/static checks; not directly used here.
with suppress(Exception):
    from src.replay_buffer import ReplayBuffer  # noqa: F401


def _select_device(device_pref: str) -> str:
    """Resolve device preference, maintaining backward compatibility."""
    cuda_ok = torch is not None and torch.cuda.is_available()
    if device_pref in {"auto", "cuda"}:
        return "cuda" if cuda_ok else "cpu"
    return "cpu"


def _find_latest_checkpoint(dir_path: str) -> Optional[str]:
    """
    Find the newest checkpoint file by modification time within the given directory.
    Supports common extensions.
    """
    if not os.path.isdir(dir_path):
        return None
    patterns = ["*.pt", "*.pth", "*.ckpt", "*.tar"]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(dir_path, pat)))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _tuple2(values: list[str]) -> Tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two values, e.g. --puck-speed-init 4.0 8.0")
    try:
        return float(values[0]), float(values[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirHockey DQN self-play trainer")

    # Environment geometry and physics
    parser.add_argument("--width", type=int, help="Table width in pixels")
    parser.add_argument("--height", type=int, help="Table height in pixels")
    parser.add_argument("--dt", type=float, help="Seconds per physics step (simulation dt)")
    parser.add_argument("--friction", type=float, help="Puck velocity friction multiplier (0,1]")
    parser.add_argument("--puck-radius", type=float, dest="puck_radius", help="Puck radius in pixels")
    parser.add_argument("--puck-mass", type=float, dest="puck_mass", help="Puck mass (arbitrary units)")
    parser.add_argument("--mallet-radius", type=float, dest="mallet_radius", help="Mallet radius in pixels")
    parser.add_argument("--mallet-speed", type=float, dest="mallet_speed", help="Mallet max speed per tick (px)")
    parser.add_argument("--mallets-per-side", type=int, dest="mallets_per_side", help="Number of mallets each side controls (1-4)")
    parser.add_argument("--puck-speed-init", nargs=2, metavar=("MIN","MAX"), help="Initial puck speed range (px/tick)")

    # Episode and rewards
    parser.add_argument("--max-steps", type=int, dest="max_steps", help="Max steps per episode")
    parser.add_argument("--goal-height-ratio", type=float, dest="goal_height_ratio", help="Goal opening height as a fraction of table height (e.g., 0.35)")
    parser.add_argument("--reward-goal", type=float, dest="reward_goal", help="Reward for scoring a goal")
    parser.add_argument("--reward-concede", type=float, dest="reward_concede", help="Penalty for conceding a goal")
    parser.add_argument("--reward-time-penalty", type=float, dest="reward_time_penalty", help="Per-step time penalty")
    parser.add_argument("--reward-toward-opponent", type=float, dest="reward_toward_opponent", help="Small reward for puck moving toward opponent")
    parser.add_argument("--reward-distance-weight", type=float, dest="reward_distance_weight", help="Weight for negative distance shaping")
    parser.add_argument("--reward-on-hit", type=float, dest="reward_on_hit", help="Reward for hitting the puck")

    # Observation
    parser.add_argument("--vel-norm-puck", type=float, dest="vel_norm_puck", help="Normalization constant for puck velocities")
    parser.add_argument("--mirror-right-obs", action=argparse.BooleanOptionalAction, dest="mirror_right_obs", help="Mirror observations for right agent")

    # Algorithm selection
    parser.add_argument("--algo", type=str, choices=["dqn", "dueling"], help="RL algorithm to use (dqn or dueling)")

    # DQN / RL hyperparameters
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size")
    parser.add_argument("--buffer-capacity", type=int, dest="buffer_capacity", help="Replay buffer capacity")
    parser.add_argument("--learn-start", type=int, dest="learn_start", help="Steps before learning starts")
    parser.add_argument("--target-sync", type=int, dest="target_sync", help="Gradient steps between target syncs")

    # Epsilon-greedy
    parser.add_argument("--eps-start", type=float, dest="eps_start", help="Initial epsilon")
    parser.add_argument("--eps-end", type=float, dest="eps_end", help="Final epsilon")
    parser.add_argument("--eps-decay-frames", type=int, dest="eps_decay_frames", help="Frames to decay epsilon")
    parser.add_argument("--reset-eps", action="store_true", dest="reset_eps", help="Reset epsilon schedule after loading checkpoint (sets frame_idx=0 and epsilon=eps_start)")
    parser.add_argument("--keep-eps-progress", action="store_true", dest="keep_eps_progress", help="When overriding epsilon params, keep prior decay progress instead of restarting schedule")

    # Training control
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], help="Compute device preference")

    # Rendering
    parser.add_argument("--render-fps", type=int, dest="render_fps", help="Renderer FPS on visualize episodes")
    parser.add_argument("--visualize-every-n", type=int, dest="visualize_every_n", help="Visualize every N episodes (0 disables)")
    parser.add_argument("--no-sound", action=argparse.BooleanOptionalAction, dest="no_sound", help="Disable audio (SDL dummy)")

    # Checkpointing
    parser.add_argument("--checkpoint-every", type=int, dest="checkpoint_every", help="Episodes between checkpoints (0 disables)")
    parser.add_argument("--checkpoint-dir", type=str, dest="checkpoint_dir", help="Directory to store checkpoints")
    parser.add_argument("--load-latest", action=argparse.BooleanOptionalAction, dest="load_latest", help="Load latest checkpoint on startup")

    # Utility
    parser.add_argument("--print-config", action="store_true", help="Print the final config and exit")

    return parser.parse_args(argv)


def _collect_overrides(ns: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    # Only allow keys that exist on Config to avoid passing unknown args (e.g., reset_eps)
    config_field_names = {f.name for f in dataclass_fields(Config)}
    for k, v in vars(ns).items():
        if k == "print_config":
            continue
        if v is None:
            continue
        if k == "puck_speed_init":
            # Already as list[str] when present; convert to tuple[float,float]
            if isinstance(v, (list, tuple)):
                with suppress(Exception):
                    overrides[k] = (float(v[0]), float(v[1]))
            continue
        if k in config_field_names:
            overrides[k] = v
    return overrides


def main() -> None:
    """
    Entry point: builds config, environment, agents, renderer, and trainer; then starts training.
    Handles deterministic seeding, device selection, and optional checkpoint loading.
    """
    # Ensure no audio is initialized by pygame
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # Parse CLI and build configuration overrides
    args = parse_args(sys.argv[1:])
    overrides = _collect_overrides(args)

    # Instantiate configuration with overrides
    config = Config(**overrides)

    if getattr(args, "print_config", False):
        # Print resolved config and exit
        from pprint import pprint

        print("Final configuration (resolved):")
        d = config.to_dict()
        pprint(d)
        return

    # Device selection (use resolved device from Config)
    device_str = getattr(config, "device_str", "cpu")

    # Seed everything for determinism
    seed_everything(getattr(config, "seed", 42))

    # Ensure checkpoint directory exists
    ckpt_dir = getattr(config, "checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build environment
    env = AirHockeyEnv(config)

    

    # Peek at observation dimension by resetting once (trainer will reset again per episode)
    try:
        obs_left, obs_right = env.reset(seed=getattr(config, "seed", None))
        obs_dim = len(obs_left)
    except Exception:
        # Use the calculated obs_dim from config
        obs_dim = config.obs_dim

    action_space_n = config.action_space_n  # Dynamic based on mallets_per_side

    # Build agents
    agent_kwargs = dict(
        obs_dim=obs_dim,
        action_space_n=action_space_n,
        lr=getattr(config, "lr", 1e-4),
        gamma=getattr(config, "gamma", 0.99),
        batch_size=getattr(config, "batch_size", 128),
        buffer_capacity=getattr(config, "buffer_capacity", 100_000),
        learn_start=getattr(config, "learn_start", 5_000),
        target_sync=getattr(config, "target_sync", 1_000),
        eps_start=getattr(config, "eps_start", 1.0),
        eps_end=getattr(config, "eps_end", 0.05),
        eps_decay_frames=getattr(config, "eps_decay_frames", 100_000),
        device=device_str,
    )

    # Select agent class based on algorithm
    algo = getattr(config, "algo", "dqn").lower()
    if algo == "dueling":
        from src.agents.dueling_dqn import DuelingDQNAgent as AgentClass
    else:
        from src.agents.dqn import DQNAgent as AgentClass

    try:
        agent_left = AgentClass(**agent_kwargs)
        agent_right = AgentClass(**agent_kwargs)
    except TypeError:
        # Fallback to minimal constructor if the above signature doesn't match
        agent_left = AgentClass(obs_dim=obs_dim, action_space_n=action_space_n, device=device_str)
        agent_right = AgentClass(obs_dim=obs_dim, action_space_n=action_space_n, device=device_str)

    # Optional: load latest checkpoint
    if getattr(config, "load_latest", True):
        if latest := _find_latest_checkpoint(ckpt_dir):
            try:
                print(f"[main] Loading latest checkpoint: {latest}")
                load_checkpoint(agent_left, agent_right, latest)
                # After loading, apply any explicit epsilon overrides from CLI so they take precedence
                def _apply_eps_overrides(agent: DQNAgent) -> None:
                    if "eps_start" in overrides and overrides["eps_start"] is not None:
                        try:
                            agent.eps_start = float(overrides["eps_start"])
                            agent.epsilon = float(overrides["eps_start"])  # force current epsilon
                        except Exception:
                            pass
                    if "eps_end" in overrides and overrides["eps_end"] is not None:
                        with suppress(Exception):
                            agent.epsilon_min = float(overrides["eps_end"])
                    if "eps_decay_frames" in overrides and overrides["eps_decay_frames"] is not None:
                        with suppress(Exception):
                            agent.epsilon_decay_steps = int(overrides["eps_decay_frames"])
                    # Determine if we should restart schedule
                    restart = False
                    if getattr(args, "reset_eps", False):
                        restart = True
                    elif ("eps_start" in overrides and overrides["eps_start"] is not None and not getattr(args, "keep_eps_progress", False)):
                        # Auto-restart when user explicitly sets a new start unless they opt to keep progress
                        restart = True
                    if restart:
                        agent.frame_idx = 0
                        # If eps_start override existed we already set epsilon above; else sync to existing start
                        if "eps_start" not in overrides or overrides["eps_start"] is None:
                            agent.epsilon = agent.eps_start

                _apply_eps_overrides(agent_left)
                _apply_eps_overrides(agent_right)
            except Exception:
                print(f"[main] Failed to load checkpoint: {latest}")
                traceback.print_exc()
        else:
            print(f"[main] No checkpoint found in {ckpt_dir}; starting fresh.")

    # Build renderer
    try:
        renderer = Renderer(
            width=getattr(config, "width", 800),
            height=getattr(config, "height", 400),
            fps=getattr(config, "render_fps", 60),
        )
    except TypeError:
        # Fallback if fps named differently or not supported
        renderer = Renderer(
            width=getattr(config, "width", 800),
            height=getattr(config, "height", 400),
        )

    # Build trainer
    try:
        trainer = Trainer(
            env=env,
            agent_left=agent_left,
            agent_right=agent_right,
            renderer=renderer,
            visualize_every_n=getattr(config, "visualize_every_n", 5),
            device=device_str,
            checkpoint_every=config.checkpoint_every,
        )
    except TypeError:
        # Fallback with minimal args
        trainer = Trainer(env, agent_left, agent_right, renderer)

    episodes = int(getattr(config, "episodes", 10_000))
    print(
        f"[main] Starting training: episodes={episodes}, device={device_str}, "
        f"render_every_n={getattr(config, 'visualize_every_n', 5)}"
    )

    try:
        trainer.run(episodes)
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user. Shutting down gracefully...")
    except Exception:
        print("[main] Unhandled exception during training:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure renderer resources are freed
        with suppress(Exception):
            renderer.close()

    print("[main] Training finished.")


if __name__ == "__main__":
    main()