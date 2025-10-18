### The Problem: Sparse Rewards in Large Grids

The original reward structure was designed for small grids and became problematic for larger environments:

**Previous Reward Function (Fixed Values):**
- **Goal reward**: `1.0` (fixed, regardless of grid size)
- **Step penalty**: `-0.005` per step (fixed)
- **Timeout penalty**: `-0.02` (fixed)

**Why This Failed for Large Grids:**
The sparse reward structure became problematic starting around **30x30 grids** and severely impacted learning in larger environments:

- **50x50 grid example**: Optimal path = 99 steps
  - Optimal total reward: `1.0 + 99 × (-0.005) = 0.505`
  - Worst case (5000 steps): `1.0 + 5000 × (-0.005) = -24.0`
  - The goal reward was completely inadequate!

- **Problem**: The fixed goal reward of `1.0` was insufficient to overcome the cumulative step penalties in larger grids, making the agent learn that reaching the goal wasn't worth the effort.

### The Solution: Grid-Size-Aware Reward Scaling

**Previous Reward Function (Mathematical Scaling):**

1. **Scaled Goal Reward**:
   ```python
   max_step_penalty = max_steps_per_episode * 0.01
   goal_reward = max(10.0, 2.0 * max_step_penalty)
   ```
   - Ensures goal reward is at least 2x the maximum possible step penalty
   - Scales with `max_steps_per_episode` to maintain proper incentives

2. **Scaled Step Penalty**:
   ```python
   base_penalty = 0.01
   grid_area = grid_size_x * grid_size_y
   scaled_penalty = base_penalty * (100.0 / grid_area) ** 0.5
   ```
   - Scales inversely with grid area (square root scaling)
   - Larger grids get smaller per-step penalties to prevent over-penalization

3. **Scaled Timeout Penalty**:
   ```python
   timeout_penalty = max(0.1, 0.01 * (grid_size_x + grid_size_y))
   ```
   - Scales with grid perimeter
   - Maintains appropriate penalty relative to grid size

### Current Implementation: Tiered Reward System

**Latest Reward Function (Tiered Approach):**

The current implementation uses a **tiered reward system** based on grid area thresholds:

```python
grid_area = grid_size_x * grid_size_y

if grid_area <= 100:  # Small grids (10x10 and smaller)
    goal_reward = 10.0
    step_penalty = 0.1
    wall_penalty = 0.5
    timeout_penalty = 1.0
elif grid_area <= 2500:  # Medium grids (up to 50x50)
    goal_reward = 50.0
    step_penalty = 0.05
    wall_penalty = 1.0
    timeout_penalty = 5.0
else:  # Large grids (100x100+)
    goal_reward = 100.0
    step_penalty = 0.02
    wall_penalty = 0.1
    timeout_penalty = 10.0
```

**Key Features:**
- **Tiered scaling**: Three distinct reward tiers based on grid area
- **Wall hit penalties**: Explicit penalties for hitting walls/obstacles
- **Timeout penalties**: Separate penalties for episode timeouts
- **Simplified logic**: Easier to understand and debug than mathematical scaling

### Results: Current Tiered Reward System

| Grid Size | Tier | Step Penalty | Wall Penalty | Goal Reward | Timeout Penalty | Optimal Total* |
|-----------|------|--------------|--------------|-------------|-----------------|----------------|
| 5x5       | Small| 0.1          | 0.5          | 10.0        | 1.0            | 9.0             |
| 10x10     | Small| 0.1          | 0.5          | 10.0        | 1.0            | 8.0             |
| 20x20     | Mediu| 0.05        | 1.0          | 50.0        | 5.0            | 48.0            |
| 50x50     | Mediu| 0.05        | 1.0          | 50.0        | 5.0            | 45.0            |
| 100x100   | Large| 0.02         | 0.1          | 100.0       | 10.0           | 98.0            |

*Optimal Total = Goal Reward - (Optimal Steps × Step Penalty)

For implementation details, see the `step()` function in `gridworld2d.py`.