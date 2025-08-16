# AirHockey Trainer

A self-play RL trainer for Air Hockey with support for multiple mallets per side.

## Multi-Mallet Feature

Each side can now control multiple mallets simultaneously! Use `--mallets-per-side N` where N is between 1-4.

- **Action Space**: With N mallets per side, the action space becomes 5^N (each mallet has 5 actions: stay, up, down, left, right)
- **Observation Space**: Dynamically sized based on mallet count: 4 (puck) + N*4*2 (all mallets from both sides)
- **Positioning**: Mallets are evenly distributed vertically on their respective sides
- **Rendering**: Multiple mallets are shown with color variations (different intensities of blue/red)

Examples:
- 1 mallet per side: 5 actions, 12 observations (original behavior)  
- 2 mallets per side: 25 actions, 20 observations
- 3 mallets per side: 125 actions, 28 observations
- 4 mallets per side: 625 actions, 36 observations

## Algorithm selection

Use `--algo` to choose the RL algorithm:
- `dqn` (default): Standard Double DQN
- `dueling`: Dueling Double DQN (value/advantage streams)

Example:
```bash
python ./main.py --algo dueling --episodes 100 --visualize-every-n 0
```

## Hyperparameter search

Experiment with different training settings using a simple random search
utility:

```bash
python -m src.hparam_opt --trials 5 --episodes 5
```

This runs a few short self-play sessions and prints the configuration that
achieved the highest average reward for the left agent.

## CLI usage

You can override any `Config` field from the command line via `main.py`.

Examples:

- Run 100 episodes on CPU and disable visualization:

```powershell
pwsh
python .\main.py --episodes 100 --device cpu --visualize-every-n 0
```

- Change table size and mallet speed, print the final resolved config and exit:

```powershell
python .\main.py --width 1024 --height 512 --mallet-speed 10 --print-config
```

- Use multiple mallets per side (2 mallets each):

```powershell
python .\main.py --mallets-per-side 2 --visualize-every-n 1
```

- Adjust learning rate and epsilon schedule, and set checkpoint directory:

```powershell
python .\main.py --lr 3e-4 --eps-start 1.0 --eps-end 0.1 --eps-decay-frames 150000 --checkpoint-dir .\checkpoints
```

Common flags (subset):

- `--episodes` Number of training episodes
- `--device` one of `auto|cpu|cuda|mps`
- `--visualize-every-n` Visualize every N episodes (0 disables)
- `--checkpoint-every` Save checkpoint every N episodes (0 disables)
- `--checkpoint-dir` Directory for checkpoints
- `--width`/`--height` Table size in pixels
- `--mallet-speed` Max mallet speed per step
- `--mallets-per-side` Number of mallets each side controls (1-4, default: 2)
- `--lr`, `--gamma`, `--batch-size`, `--buffer-capacity`, `--target-sync`
- `--eps-start`, `--eps-end`, `--eps-decay-frames`

Use `--print-config` to see the final resolved configuration (including resolved device).

