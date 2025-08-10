# AirHockey Trainer

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
- `--lr`, `--gamma`, `--batch-size`, `--buffer-capacity`, `--target-sync`
- `--eps-start`, `--eps-end`, `--eps-decay-frames`

Use `--print-config` to see the final resolved configuration (including resolved device).

