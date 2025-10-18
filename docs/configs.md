## Configuration

You can modify the following parameters in `main.py` (tabular Q-learning) or `main_nn.py` (Deep Q-Network):

### Environment Parameters

- `grid_size_x`: Width of the 2D grid (default: 20 for tabular, 7 for DQN)
- `grid_size_y`: Height of the 2D grid (default: 20 for tabular, 7 for DQN)
- `start_pos`: Starting position (default: (0, 0))
- `goal_pos`: Goal position (default: (grid_size_x-1, grid_size_y-1))

### Obstacle Configuration (DQN only)

- `USE_OBSTACLES`: Enable/disable obstacles in the environment (default: True)
- `OBSTACLE_DENSITY`: Fraction of cells to fill with obstacles, 0.0 to 1.0 (default: 0.35)

### Training Parameters

- `num_episodes`: Number of training episodes (default: 10000 for both). I am able to do this safely in main_nn.py because I am now detecting training to be complete based on the condition that the min_steps possible to traverse the grid have been achieved in the last 5 runs. See LEARNING_ACHIEVED_THRESHOLD below.
- `learning_rate`: Learning rate for Q-value updates (default: 0.1 for tabular, 0.001 for DQN)
- `discount_factor`: Discount factor for future rewards (default: 0.99)
- `epsilon`: Initial exploration rate (default: 1.0)
- `epsilon_decay`: Decay rate for exploration (default: 0.99 for tabular, 0.999 for DQN)
- `epsilon_min`: Minimum exploration rate (default: 0.01)
- `exploration_strategy`: Exploration strategy (default: "epsilon_greedy")

### Learning Detection Parameters

- `LEARNING_ACHIEVED_THRESHOLD`: Number of consecutive optimal episodes to confirm learning (default: 5)

### DQN-Specific Parameters (main_nn.py)

**Note**: The following parameters are automatically computed based on grid size using adaptive sizing algorithms:

- `hidden_size`: Neural network hidden layer size (auto-computed: 128-2048)
- `buffer_size`: Experience replay buffer size (auto-computed: 5,000-200,000)
- `batch_size`: Training batch size (auto-computed: 28-256)
- `target_update_freq`: Target network update frequency (default: 100)

**Adaptive Sizing Details**:
- **Hidden Size**: Scales with grid area using target parameters per state (12-25 params/state)
- **Buffer Size**: Scales with square root of grid area for diverse experiences
- **Batch Size**: Scales with grid area to power 0.25 for stable training

**Manual Override**: If you want to manually set these parameters, you can modify the computed values after the adaptive sizing functions in `main_nn.py`.

### Display and Visualization Parameters

- `SHOW_PLOTS`: Enable/disable matplotlib visualizations (default: False)
- `SHOW_GRID_DISPLAYS`: Show grid displays in terminal (auto-disabled for grids > 1000 states)
- `LIVE_REFRESH_PER_SECOND`: Terminal UI refresh frequency (default: 1 for DQN, 50 for tabular)
- `DISPLAY_STEP_INTERVAL`: Steps between NN-driven display updates (default: 10000, set to 0 to disable)
- `sleep_time`: Time to sleep between episodes (default: 0)

### WandB Logging Parameters

- `USE_WANDB`: Enable/disable WandB logging (default: False)
- `N_IMAGE_EPISODES`: Number of intermediate episodes to log with images (default: 10)

### Episode Step Cap

To prevent episodes from wandering too long under high exploration, the environment enforces a `max_steps_per_episode` cap:

- In the environment (`gridworld2d.py`): `GridWorld2D` accepts `max_steps_per_episode` and terminates the episode with a small timeout penalty when exceeded.
- In the tabular script (`main.py`): `max_steps_per_episode` is computed by `compute_max_steps(grid_x, grid_y, epsilon)` which scales with Manhattan distance and grid area, with an optional adaptive mode based on recent episode lengths.
- In the DQN script (`main_nn.py`): Same adaptive computation as tabular, with additional obstacle-aware pathfinding.

**Adaptive Step Cap Formula**:
- Base cap: `D * (k0 + k_eps * epsilon)` where D = Manhattan distance
- Adaptive cap: Uses 75th percentile of recent episode lengths
- Bounds: Between `2*D` and `2*area` with maximum limit of `30*(grid_x + grid_y)`

Display note: The steps summary shows entries like `42(M)` in "Steps in last 10 episodes:" when an episode ends by hitting the max step cap.

### Device Selection (DQN only)

The DQN automatically selects the best available device:
- Apple Silicon (MPS): `torch.device("mps")` --> this is the only one I have tested this on. 
- CUDA: `torch.device("cuda")`
- CPU: `torch.device("cpu")`

### Performance Notes

- **Tabular Q-Learning**: Best for small grids (≤ 20x20), fast convergence
- **Deep Q-Network**: Best for large grids (≥ 50x50), scales automatically with grid size
- **Adaptive Sizing**: DQN parameters automatically scale to maintain optimal learning capacity
- **Early Stopping**: Training stops automatically when learning is achieved (5 consecutive optimal episodes)