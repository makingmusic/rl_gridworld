# AGENTS.md

Instructions for AI coding agents (Claude Code, Cursor, Codex, etc.) working in this repository.

This file is the source of truth for "how to be useful here". If you're a human contributor, you can read this too — but it's written for agents.

---

## What this project is

A learn-in-public Reinforcement Learning sandbox. It implements **Q-learning** in a 2D gridworld, in two flavors:

- **Tabular Q-learning** — `main.py` + `q_agent2D.py`
- **Deep Q-Network (DQN)** — `main_nn.py` + `q_agent_nn.py`

The environment lives in `gridworld2d.py`. Visualization is in `plots.py`. Optional W&B logging is in `logWandB.py`.

There is also a sibling experiment in **`balance/`** — classical control (PID) on balancing problems (inverted pendulum now, cartpole later). Its env interface mirrors `GridWorld2D` so an RL agent can plug in later. It uses the same `myenv/` venv and dependencies. See `balance/README.md` and `plans/balance.md`.

It is a **personal learning project**, not production code. The author values clarity and incremental, understandable changes over architectural purity. Treat this like a lab notebook with running code.

Read `README.md` for the author's narrative context. Read `docs/` for deeper topic guides.

---

## Read these before changing anything

| If you are touching... | Read first |
|---|---|
| Hyperparameters or config | `docs/configs.md` |
| The neural network / DQN | `docs/nn.md` |
| Reward shaping or `step()` | `docs/rewards.md` |
| The environment / obstacles | `docs/environment.md` |
| Tabular Q-learning agent | `docs/tabular.md` |
| Module layout, who calls who | `docs/architecture.md` |
| Setup, running, devcontainer | `docs/development.md` |
| W&B logging | `docs/wandb.md` |
| `balance/` (PID + TUI) | `balance/README.md`, `plans/balance.md` |
| Long-term ideas / roadmap | `docs/future-todo.md` |

If you're about to edit a file and the matching doc is silent or wrong, **update the doc in the same change**.

---

## Ground rules

### 1. Don't over-engineer

This is a learning project. Resist the urge to:

- Introduce abstractions, base classes, factories, or registries "for future flexibility"
- Add config layers (YAML, env vars) when the existing in-file constants work
- Split a 200-line file into five 40-line files
- Add type hints everywhere if the surrounding code doesn't have them
- Refactor working code that wasn't part of the request

A bug fix is a bug fix. A new feature is a new feature. Don't sneak refactors in.

### 2. Keep both implementations in sync conceptually

Tabular (`main.py`) and DQN (`main_nn.py`) intentionally mirror each other where it makes sense:

- Same hyperparameter names (`learning_rate`, `epsilon_decay`, etc.) even when values differ
- Same early-stopping behavior (`LEARNING_ACHIEVED_THRESHOLD`)
- Same display/visualization structure

If you add a meaningful capability to one (e.g., a new metric, a new reward term, a new display panel), consider whether the other should get it too. Don't blindly duplicate, but don't silently diverge either. Call out the divergence in your response if you choose not to mirror.

### 3. Respect the configuration style

All hyperparameters live as **module-level constants at the top of `main.py` / `main_nn.py`**. That's deliberate — the author wants every knob visible in one scroll.

- Do **not** move config to YAML, argparse, or pydantic
- Do **not** introduce a `Config` dataclass that wraps these constants
- If you add a new tunable, put it at the top of the file with the others, give it a clear name, and add a one-line comment if its meaning isn't obvious

### 4. Don't add dependencies casually

`pyproject.toml` is already heavy with pinned transitive deps. Before `uv add`-ing anything:

- Confirm it's not already installed
- Prefer stdlib or existing deps (numpy, torch, rich, matplotlib, networkx) where possible
- Ask the user before adding a new top-level dependency

### 5. Python version and runtime

The project requires **Python 3.14+** (see `pyproject.toml`). It is run via `uv`:

```bash
uv run python main.py        # tabular
uv run python main_nn.py     # DQN
```

Do not propose changes that require older Python versions or that bypass `uv`.

### 6. Device selection (DQN)

The DQN auto-picks MPS → CUDA → CPU. The author has tested **MPS only**. If you change device logic, preserve this fallback order and don't assume CUDA works.

### 7. Be careful with `run.sh`

`run.sh` is the canonical setup entry point. It's idempotent. If you change it, make sure it still:

- Works on a clean clone
- Doesn't activate/leak into the user's outer virtualenv
- Doesn't require interactive input
- Still prints the final "to run, do X" hint

---

## Working with this codebase

### Where things live

```
.
├── main.py              # Tabular Q-learning entry point + config
├── main_nn.py           # DQN entry point + config
├── q_agent2D.py         # Tabular Q-learning agent
├── q_agent_nn.py        # DQN agent (DQN, ExperienceReplay, DQNAgent)
├── gridworld2d.py       # Environment + maze generation + A*
├── plots.py             # Rich terminal UI + matplotlib visualizations
├── logWandB.py          # W&B logging wrapper
├── run.sh               # Setup script (uv-based)
├── pyproject.toml       # Python 3.14+, deps
├── README.md            # Narrative project intro
├── AGENTS.md            # This file
├── docs/                # Topic guides — see table above
└── .devcontainer/       # VS Code / Cursor devcontainer config
```

See `docs/architecture.md` for a deeper map.

### Running

```bash
./run.sh                      # First-time setup (idempotent)
uv run python main.py         # Tabular
uv run python main_nn.py      # DQN
```

A full training run is short — tabular finishes in seconds on small grids; DQN takes longer but uses early stopping. **You can and should run the code to verify your changes**, especially for anything touching the training loop, rewards, or environment.

### Verifying changes

This project has no test suite. To validate a change:

1. **Run it.** `uv run python main.py` and `uv run python main_nn.py` both finish quickly on default grids.
2. **Watch the terminal UI.** The Rich display shows convergence, optimal-path detection, and step counts. If learning regresses (more steps, no convergence, learning-achieved not firing), your change broke something.
3. **For environment changes:** sanity-check with `USE_OBSTACLES = False` first to isolate from maze-generation noise.
4. **For reward changes:** read `docs/rewards.md` and confirm the tier math still adds up.

If you cannot run the code (sandboxed agent), say so explicitly. Don't claim "tested" when you didn't.

---

## Commit / PR conventions

- Commit messages are short, lowercase, descriptive. Look at `git log` for style.
- One logical change per commit when possible.
- Update the relevant `docs/*.md` in the **same commit** as the code change. Stale docs are worse than no docs here.
- Do **not** commit unless the user explicitly asks. Show the diff first.

---

## What to do when something is ambiguous

Ask. Don't guess on:

- Whether a refactor is in scope
- Whether to mirror a change across tabular and DQN
- Whether to add a new dependency
- Whether to commit / push
- Whether to delete code that looks unused

A one-line clarification beats a 200-line PR that has to be redone.

---

## Things specifically not to do

- ❌ Don't add a `tests/` directory or pytest setup unless asked
- ❌ Don't add CI workflows
- ❌ Don't add type-stub generation or `mypy.ini`
- ❌ Don't reformat the whole repo with black / ruff in a single sweep
- ❌ Don't introduce logging frameworks (the project uses Rich + print + W&B)
- ❌ Don't move constants into a `constants.py` module
- ❌ Don't replace the in-file config block with argparse, click, or hydra
- ❌ Don't delete `logWandB.py` or the W&B code path even though it's disabled by default — the author may turn it back on

If you genuinely think one of these is the right call, **say why in your response and wait for approval** before doing it.
