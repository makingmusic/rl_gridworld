# Tabular Q-Learning

The "classic" implementation. Entry point: `main.py`. Agent: `q_agent2D.QLearningAgent2D`.

For the DQN counterpart, see [nn.md](nn.md). For the cross-cutting reward / step-cap / config details, see [configs.md](configs.md) and [rewards.md](rewards.md).

## Why tabular first

The tabular agent exists because it's the simplest thing that works and the easiest thing to debug. If something seems off in the DQN, the first sanity check is "does the tabular agent learn this grid?" — if yes, the env is fine and the issue is in the neural network.

## The agent

`QLearningAgent2D` stores Q-values in a plain Python dict:

```python
self.q_table: dict[tuple[int, int], np.ndarray] = {}
```

Each key is a `(x, y)` state. Each value is a length-4 array of Q-values, one per action (`up`, `down`, `left`, `right`). States that have never been visited are returned as zeros via `defaultdict`-style access.

### Action selection (ε-greedy)

With probability `epsilon`: random action. Otherwise: `argmax` over the Q-values for the current state. Ties broken by `np.argmax`'s first-occurrence rule.

### Update rule

Standard Bellman:

```
Q(s, a) ← Q(s, a) + α · (r + γ · max_a' Q(s', a') − Q(s, a))
```

with:

- `α` = `learning_rate` (default `0.1`)
- `γ` = `discount_factor` (default `0.99`)
- terminal states get `max_a' Q(s', a') = 0`

The update happens **every step** — no replay buffer, no batching.

### Epsilon decay

After each episode:

```
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

Defaults: start at `1.0`, decay `0.99`, floor `0.01`.

## The training loop (`main.py`)

In order:

1. Build the env (`GridWorld2D`) and agent (`QLearningAgent2D`).
2. Pre-compute `max_steps_per_episode` via `compute_max_steps(...)`.
3. For each episode:
   - Reset env, accumulate steps & reward.
   - Step → update → repeat until `done`.
   - Decay epsilon.
   - Update the Rich live display (policy arrows, episode counter, current path).
   - Check learning detection: if the last `LEARNING_ACHIEVED_THRESHOLD` episodes all hit `optimal_steps`, exit early.
4. After the loop, optionally render matplotlib plots and save the final Q-table image.

The whole loop is intentionally flat — one for-loop, visible Bellman update, no abstraction between the data and the math. This is on purpose; see [AGENTS.md](../AGENTS.md).

## Defaults (tabular)

| Knob | Default |
|---|---|
| `grid_size_x` | 20 |
| `grid_size_y` | 20 |
| `num_episodes` | 10000 |
| `learning_rate` | 0.1 |
| `discount_factor` | 0.99 |
| `epsilon` (initial) | 1.0 |
| `epsilon_decay` | 0.99 |
| `epsilon_min` | 0.01 |
| `LEARNING_ACHIEVED_THRESHOLD` | 5 |
| `USE_WANDB` | False |
| `SHOW_PLOTS` | False |

Full list in [configs.md](configs.md).

## Strengths and limits

**Good for:**
- Small grids (≤ 20×20), where the Q-table fits comfortably in memory.
- Sanity-checking the environment and reward shaping.
- Verifying convergence — tabular Q-learning is guaranteed to find the optimal policy given enough exploration.

**Bad for:**
- Large grids — memory is `O(states × 4)`, and visiting every state often enough takes more episodes than is patient.
- Obstacle navigation — the tabular path doesn't currently enable obstacles, and even if it did, sparse exploration of large mazes is slow.
- Generalization — tabular Q-learning has none. A new start position is a fresh problem.

When you hit these limits, switch to the DQN path.

## Visualization

The same `plots.py` helpers are used for both agents. `getQTable()` on the tabular agent just returns the dict; `plots.py` queries it for each cell to draw arrows. `display_actual_path()` greedily rolls out from start to show the current best path.

## Debugging checklist

If the tabular agent isn't learning:

1. Is the goal actually reachable from the start? (Trivially yes without obstacles.)
2. Is `max_steps_per_episode` large enough? Under high `epsilon`, the cap can clip even the optimal trajectory.
3. Is `epsilon_decay` too aggressive? Too-fast decay locks in a suboptimal greedy policy.
4. Are reward magnitudes balanced for the grid size? See [rewards.md](rewards.md).
5. Is `LEARNING_ACHIEVED_THRESHOLD` set sensibly? Setting it to `1` will declare victory on the first lucky episode.
