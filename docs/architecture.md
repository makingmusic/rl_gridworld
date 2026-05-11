# Architecture

A map of the codebase: what each file does, how they fit together, and the data flow during a training run.

## Module map

```
┌───────────────────────┐         ┌───────────────────────┐
│      main.py          │         │     main_nn.py        │
│  (tabular entrypoint) │         │   (DQN entrypoint)    │
│  - config constants   │         │  - config constants   │
│  - training loop      │         │  - adaptive sizing    │
│  - Rich live UI       │         │  - training loop      │
└──────────┬────────────┘         └──────────┬────────────┘
           │                                  │
           │    ┌─────────────────────┐       │
           ├───▶│   gridworld2d.py    │◀──────┤
           │    │   - GridWorld2D     │       │
           │    │   - step / reset    │       │
           │    │   - reward tiers    │       │
           │    │   - obstacles, A*   │       │
           │    │   - maze generator  │       │
           │    └─────────────────────┘       │
           │                                  │
┌──────────▼──────────┐         ┌─────────────▼─────────┐
│   q_agent2D.py      │         │    q_agent_nn.py      │
│  QLearningAgent2D   │         │  - DQN (nn.Module)    │
│  - Q-table dict     │         │  - ExperienceReplay   │
│  - epsilon-greedy   │         │  - DQNAgent           │
│  - tabular update   │         │  - target network     │
└──────────┬──────────┘         └─────────────┬─────────┘
           │                                  │
           │       ┌────────────────┐         │
           ├──────▶│   plots.py     │◀────────┤
           │       │  - Rich grid   │         │
           │       │  - matplotlib  │         │
           │       │  - A* viz      │         │
           │       └────────────────┘         │
           │                                  │
           │       ┌────────────────┐         │
           └──────▶│  logWandB.py   │◀────────┘
                   │  - optional    │
                   └────────────────┘
```

## File-by-file

### Entry points

**`main.py`** — Tabular Q-learning. Top of file declares all hyperparameters as module-level constants. Builds a `GridWorld2D`, a `QLearningAgent2D`, then runs the episode loop. The Rich `Live` display is composed in-line.

**`main_nn.py`** — DQN. Same shape as `main.py` but adds:
- `compute_optimal_nn_size`, `compute_adaptive_buffer_size`, `compute_adaptive_batch_size` — pick hyperparameters from grid size before the agent is created.
- A `max_steps_per_episode` cap that's adaptive (see [configs.md](configs.md)).
- Obstacle generation via `generate_solvable_maze` and A* optimal-path length for learning detection.

The two entry points intentionally mirror each other. Agents working here should preserve that symmetry unless there's a reason not to (see [AGENTS.md](../AGENTS.md)).

### Environment

**`gridworld2d.py`** — Pure-Python environment. No torch dependency.

- `GridWorld2D` — class holding grid bounds, start/goal, obstacles, and a step counter. Methods:
  - `reset()` → starting state
  - `step(action)` → `(next_state, reward, done, info)` with tiered reward shaping (see [rewards.md](rewards.md))
  - `_is_valid_position` / `_validate_position` — bounds + obstacle checks
- `a_star_pathfinding(...)` — returns optimal path length (used to detect "learning achieved").
- `is_solvable(...)` — BFS-based check used during maze generation.
- `generate_solvable_maze(...)` — places obstacles such that start→goal is reachable; rejects unsolvable layouts.

The environment knows about the **reward tiers**. Don't move reward logic out into the agents.

### Agents

**`q_agent2D.py`** — `QLearningAgent2D`. Q-values stored in a dict `{(x, y): np.ndarray(4)}`. Implements ε-greedy action selection and the standard tabular Bellman update. See [tabular.md](tabular.md).

**`q_agent_nn.py`** — Three classes:
- `DQN(nn.Module)` — feedforward MLP, 3 ReLU hidden layers. Input = normalized `(x, y)`. Output = Q-values for 4 actions.
- `ExperienceReplay` — bounded `deque` buffer with random batch sampling.
- `DQNAgent` — wraps `DQN` (online + target), `ExperienceReplay`, optimizer, ε-greedy. Exposes the same surface as `QLearningAgent2D` (`choose_action`, `update_q_value`, `getQTable`) so `plots.py` and the training loops can be agent-agnostic where possible.

See [nn.md](nn.md) for the full DQN walkthrough.

### Visualization

**`plots.py`** — All the display code:
- `create_grid_display` / `update_grid_display` — Rich `Table` showing the current policy as arrows, with start/goal/obstacles marked.
- `display_actual_path` — greedy rollout from start, rendered as a path.
- `plotStepsPerEpisode`, `plotEpsilonDecayPerEpisode`, `plotQTableValues` — matplotlib plots (gated by `SHOW_PLOTS`).
- `saveQTableAsImage` — Q-table heatmap, used for W&B image logging.
- `get_best_path_length` — greedy-policy path length, used in learning detection.
- `shouldThisEpisodeBeLogged` — decides which intermediate episodes get image logging.

### Logging

**`logWandB.py`** — Thin wrapper around `wandb.init` / `wandb.log` / `wandb.finish`. Off by default. `logEpisodeWithImageControl` decides whether to attach Q-table images based on episode index. See [wandb.md](wandb.md).

## Data flow during a training run

1. **Setup** — entry script reads config constants, instantiates `GridWorld2D` (with obstacles for DQN), then the agent.
2. **Episode start** — `env.reset()` returns the start state.
3. **Step loop** — agent chooses action (ε-greedy), env returns `(next_state, reward, done)`, agent updates Q-values (or stores in replay buffer + trains for DQN).
4. **Episode end** — entry script records steps + reward, decays ε, updates the Rich display, optionally logs to W&B.
5. **Learning detection** — if `steps == optimal_path_length` (A* for DQN, Manhattan distance for obstacle-free tabular) for `LEARNING_ACHIEVED_THRESHOLD` consecutive episodes, training exits early.
6. **Teardown** — final plots / images saved, W&B closed.

## Cross-cutting conventions

- **State** is always a `(x, y)` tuple. (0, 0) is bottom-left.
- **Actions** are strings `"up" | "down" | "left" | "right"`, but agents internally use integer indices `0..3` matching `env.actions`.
- **Obstacles** are a `set[tuple[int, int]]`. Empty set means no obstacles.
- **Q-table for viz** — even the DQN exposes `getQTable()` so `plots.py` doesn't care which agent it's drawing.

If you're adding a new agent, follow the existing surface: `choose_action(state)`, `update_q_value(state, action, reward, next_state, done)`, `getQTable()`, `getQTableAsPolicyArrows()`.
