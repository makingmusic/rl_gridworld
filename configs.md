## Configuration

You can modify the following parameters in `main.py` (tabular Q-learning) or `main_nn.py` (Deep Q-Network):

### Environment Parameters

- `grid_size_x`: Width of the 2D grid (default: 20 for tabular, 50 for DQN)
- `grid_size_y`: Height of the 2D grid (default: 5 for tabular, 50 for DQN)
- `start_pos`: Starting position (default: (0, 0))
- `goal_pos`: Goal position (default: (grid_size_x-1, grid_size_y-1))

### Training Parameters

- `num_episodes`: Number of training episodes (default: 500 for tabular, 10000 for DQN)
- `learning_rate`: Learning rate for Q-value updates (default: 0.1 for tabular, 0.001 for DQN)
- `discount_factor`: Discount factor for future rewards (default: 0.99)
- `epsilon`: Initial exploration rate (default: 1.0)
- `epsilon_decay`: Decay rate for exploration (default: 0.99 for tabular, 0.999 for DQN)
- `epsilon_min`: Minimum exploration rate (default: 0.01)

### DQN-Specific Parameters (main_nn.py)

**Note**: The following parameters are automatically computed based on grid size using adaptive sizing algorithms:

- `hidden_size`: Neural network hidden layer size (auto-computed: 128-1024)
- `buffer_size`: Experience replay buffer size (auto-computed: 2,000-100,000)
- `batch_size`: Training batch size (auto-computed: 28-256)
- `target_update_freq`: Target network update frequency (default: 100)

**Manual Override**: If you want to manually set these parameters, you can modify the computed values after the adaptive sizing functions in `main_nn.py`.


### Episode Step Cap

To prevent episodes from wandering too long under high exploration, the environment enforces a `max_steps_per_episode` cap:

- In the environment (`gridworld2d.py`): `GridWorld2D` accepts `max_steps_per_episode` and terminates the episode with a small timeout penalty when exceeded.
- In the tabular script (`main.py`): configure `max_steps_per_episode` in the Grid Configuration section and it is passed to `GridWorld2D`.
- In the DQN script (`main_nn.py`): `max_steps_per_episode` is computed by `compute_max_steps(grid_x, grid_y, epsilon)` which scales with Manhattan distance and grid area, with an optional adaptive mode based on recent episode lengths.

Display note: The steps summary shows entries like `42(M)` in "Steps in last 10 episodes:" when an episode ends by hitting the max step cap.