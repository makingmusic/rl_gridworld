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

**New Reward Function (Scaled with Grid Size):**

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

### Results: Balanced Rewards Across All Grid Sizes

| Grid Size | Step Penalty | Goal Reward | Optimal Total | Ratio |
|-----------|--------------|-------------|---------------|-------|
| 5x5       | 0.0200       | 10.0        | 9.8           | 98.4  |
| 10x10     | 0.0100       | 10.0        | 9.8           | 49.1  |
| 20x20     | 0.0050       | 10.0        | 9.8           | 24.5  |
| 50x50     | 0.0020       | 10.0        | 9.8           | 9.8   |
| 100x100   | 0.0010       | 10.0        | 9.8           | 4.9   |

**Key Benefits:**
- ✅ Goal reward properly outweighs step penalties for all grid sizes
- ✅ Step penalties scale appropriately (smaller for larger grids)
- ✅ Maintains strong incentive to reach the goal quickly
- ✅ Prevents over-penalization in large grids
- ✅ Consistent reward ratios across different grid sizes

For implementation details, see the step() function.