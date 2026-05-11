# Environment

The `GridWorld2D` class in `gridworld2d.py` is the MDP that both agents interact with.

## Coordinate system

- The grid is `grid_size_x` wide and `grid_size_y` tall.
- **`(0, 0)` is bottom-left.** `x` increases rightward, `y` increases upward.
- A "state" is a `(x, y)` tuple of ints.
- Default start: `(0, 0)`. Default goal: `(grid_size_x - 1, grid_size_y - 1)` (top-right).

This is opposite to image / matrix conventions where `(0, 0)` is top-left. The visualization in `plots.py` flips rendering accordingly.

## Actions

```python
env.actions = ["up", "down", "left", "right"]
```

Agents use the **integer index** into this list (0..3). The mapping is fixed; don't reorder it without auditing every consumer.

Action effects:

| Action  | Δx | Δy |
|---------|----|----|
| up      |  0 | +1 |
| down    |  0 | -1 |
| left    | -1 |  0 |
| right   | +1 |  0 |

Moves that would leave the grid or land on an obstacle are **rejected** — the agent stays put and incurs a wall penalty (DQN) or step penalty (tabular).

## Episode lifecycle

```python
env = GridWorld2D(grid_size_x=20, grid_size_y=20,
                  start_pos=(0, 0), end_pos=(19, 19),
                  max_steps_per_episode=200,
                  obstacles=None)

state = env.reset()
while not done:
    action_idx = agent.choose_action(state)
    next_state, reward, done, info = env.step(action_idx)
    agent.update_q_value(state, action_idx, reward, next_state, done)
    state = next_state
```

`done` becomes `True` when:
1. The agent reaches `end_pos`, **or**
2. `steps_since_reset >= max_steps_per_episode` (timeout)

`info` carries extra signals like wall-hit flags used by the entry-script display.

## Step cap (`max_steps_per_episode`)

Episodes are capped so that under high exploration the agent doesn't wander forever. The cap is computed by the entry script (not the environment) and passed in.

See [configs.md](configs.md) for the formula. Short version:
- Base cap: `D * (k0 + k_eps * epsilon)` where `D` is Manhattan distance start→goal.
- Adaptive mode (DQN): also considers the 75th percentile of recent episode lengths.
- Hard bounds: `[2*D, 2*area]`, never above `30*(grid_x + grid_y)`.

When an episode ends by hitting the cap, the display shows it tagged `(M)` for "max steps", and a timeout penalty is applied.

## Rewards

Reward shaping is its own topic — see [rewards.md](rewards.md). Summary:

- **Goal reached** → positive `goal_reward`
- **Each step** → small `-step_penalty`
- **Hit a wall / obstacle** → larger `-wall_penalty` (DQN) or step penalty (tabular without obstacles)
- **Timeout** → `-timeout_penalty`

All four magnitudes are picked from a **tier table** keyed on `grid_area`:

| Area              | goal | step  | wall | timeout |
|-------------------|------|-------|------|---------|
| ≤ 100 (small)     | 10   | 0.1   | 0.5  | 1.0     |
| ≤ 2500 (medium)   | 50   | 0.05  | 1.0  | 5.0     |
| > 2500 (large)    | 100  | 0.02  | 0.1  | 10.0    |

Implementation: see `step()` in `gridworld2d.py`.

## Obstacles

Obstacles are a `set[tuple[int, int]]`. The env enforces:

- Start and goal positions **cannot** be obstacles (raises `ValueError`).
- An agent attempting to move into an obstacle stays put and gets penalized.

Obstacles only matter for the DQN path right now (`USE_OBSTACLES = True` in `main_nn.py`). The tabular script does not currently use obstacles, though `GridWorld2D` would accept them.

### Maze generation

`generate_solvable_maze(grid_size_x, grid_size_y, start_pos, goal_pos, density, ...)` in `gridworld2d.py`:

1. Randomly samples cells at the requested density.
2. Excludes start and goal.
3. Runs BFS (`is_solvable`) to verify start→goal connectivity.
4. Retries (up to a budget) if unsolvable.
5. Returns a `set` of obstacle cells, or raises if no solvable layout was found within the retry budget.

Default obstacle density is `0.35` (35% of cells). Higher densities reduce the chance of a solvable layout and increase generation time.

### A* pathfinding

`a_star_pathfinding(...)` returns the **length** of the optimal path from start to goal given the obstacle set. The entry scripts use this length as ground truth for the "learning achieved" check: if the agent completes the episode in exactly that many steps for `LEARNING_ACHIEVED_THRESHOLD` consecutive episodes, training stops.

Heuristic: Manhattan distance. Standard A* with a closed set. Returns `None` if unreachable — callers should treat that as a bug (the maze generator should never produce an unreachable layout).

## Extending the environment

Common extensions and where they'd plug in:

| Idea | Where to change |
|---|---|
| Diagonal moves | Add to `env.actions` + extend movement logic in `step()` |
| Stochastic transitions | Wrap `step()` action with a probability of slipping |
| Multi-step "jumps" | Either expand action space or post-process action in `step()` |
| Multiple goal states | Generalize `end_pos` to a set; check membership |
| Moving obstacles | Update `self.obstacles` on each step inside `step()` |
| Different reward terms | Edit the tier table in `step()` — keep it in the env, not the agent |

Whatever you extend, keep the agent surface (`choose_action`, `update_q_value`) untouched. The env is the right place for environment changes.
