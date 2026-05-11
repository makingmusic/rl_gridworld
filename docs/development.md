# Development Guide

How to get a working copy of `rl_gridworld` running on your machine, and how to iterate on it.

## Requirements

- **Python 3.14+** — the project recently switched and gets a noticeable speedup from PEP 703 / free-threading work. `pyproject.toml` enforces this.
- **[`uv`](https://docs.astral.sh/uv/)** — used for environment and dependency management. `run.sh` will install it for you if missing.
- A POSIX shell (bash/zsh). On Windows, use WSL or the devcontainer.

Optional:
- Apple Silicon (M-series) — DQN training uses MPS automatically.
- An NVIDIA GPU — DQN will use CUDA if available (untested by the author).
- A [Weights & Biases](https://wandb.ai) account, if you want experiment tracking (off by default).

## First-time setup

```bash
git clone <this-repo>
cd rl_gridworld
./run.sh
```

`run.sh` is idempotent. It will:

1. Deactivate any inherited virtualenv to avoid contamination
2. Install `uv` if missing
3. Create a project venv in `./myenv/` (via `UV_PROJECT_ENVIRONMENT=myenv`)
4. `uv sync` all dependencies pinned in `pyproject.toml` / `uv.lock`
5. Print run hints

## Running

```bash
uv run python main.py        # Tabular Q-learning
uv run python main_nn.py     # Deep Q-Network
```

Both scripts:
- Print a live Rich terminal UI showing episode progress, current policy, and learning detection
- Auto-stop when learning is "achieved" (default: 5 consecutive optimal episodes — see `LEARNING_ACHIEVED_THRESHOLD`)
- Use the defaults baked into the top of the script — edit those constants to change behavior

## Devcontainer (Cursor / VS Code)

`.devcontainer/devcontainer.json` is configured so that opening the repo in Cursor or VS Code launches a clean container. This isolates the runtime from your host Python — useful when the agent is making sweeping changes you don't want leaking into your global environment.

Open the folder in the editor → "Reopen in Container" → run `./run.sh` inside.

## Iterating

The development loop is intentionally tight:

1. Edit the relevant constants at the top of `main.py` or `main_nn.py`, or change an agent / environment file.
2. `uv run python main.py` (or `main_nn.py`).
3. Watch the terminal. The Rich UI shows steps-per-episode, epsilon decay, current policy as arrows, and "learning achieved" detection.
4. Repeat.

Tabular runs finish in seconds on the default 20×20 grid. DQN runs are longer (early-stopping helps) but on small grids still complete in a minute or two.

## Common configuration changes

For the full list see [configs.md](configs.md). A handful you'll hit often:

| Change | Where |
|---|---|
| Grid size | `grid_size_x`, `grid_size_y` at top of `main.py` / `main_nn.py` |
| Obstacles on/off (DQN only) | `USE_OBSTACLES`, `OBSTACLE_DENSITY` |
| Turn on plots | `SHOW_PLOTS = True` |
| Turn on W&B | `USE_WANDB = True` — also see [wandb.md](wandb.md) |
| Slow down terminal UI | lower `LIVE_REFRESH_PER_SECOND` |
| Make training stop sooner | lower `LEARNING_ACHIEVED_THRESHOLD` |

## Debugging notes

- **Training doesn't converge / no early-stop.** Check that the optimal path is actually solvable for the current grid + obstacle layout. For DQN, see `gridworld2d.a_star_pathfinding` — if it returns `None`, the maze is unsolvable.
- **Rich UI flicker.** Lower `LIVE_REFRESH_PER_SECOND`. On large grids, the grid display auto-disables above ~1000 states.
- **MPS errors on Apple Silicon.** Confirm `torch.backends.mps.is_available()`. Some ops fall back to CPU silently — that's fine.
- **"Module not found" after pulling.** Re-run `uv sync` (or `./run.sh`).
- **W&B prompts for login when disabled.** Make sure `USE_WANDB = False` is set in the file you're running; `run.sh` will warn but not block if you're not logged in.

## Project hygiene

- No test suite. Validation is "run it and watch the UI."
- No linters or formatters configured. Match the surrounding style.
- No CI. Anything that needs to pass before merge is enforced socially.

If you're an agent: see [../AGENTS.md](../AGENTS.md) for the dos and don'ts before making changes.
