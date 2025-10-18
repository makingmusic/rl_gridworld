## Weights & Biases (wandb) Integration

[Weights & Biases (wandb)](https://wandb.ai/) provides experiment tracking and visualization for machine learning projects. **By default, wandb logging is disabled** in this project.

### Setting up wandb

**Step 1: Install wandb**

The project uses `uv` for dependency management. wandb is already included in the dependencies:

```bash
# If using the run.sh script (recommended)
./run.sh

# Or manually with uv
uv sync
```

**Step 2: Create a wandb account and login**

1. Go to [wandb.ai](https://wandb.ai/) and create a free account
2. Login to wandb in your terminal:

```bash
# Using uv (recommended)
uv run wandb login

# Or with pip if not using uv
wandb login
```

3. Find your API key at [wandb.ai/authorize](https://wandb.ai/authorize)

**Step 3: Enable wandb logging**

- **By default, wandb logging is disabled** (`USE_WANDB = False` in both `main.py` and `main_nn.py`)
- To enable wandb logging, set `USE_WANDB = True` in the respective main file
- Projects are logged separately:
  - Tabular Q-learning: `rl-gridworld-qlearning`
  - Deep Q-Network: `rl-gridworld-dqn`

### Current Implementation Features

**Smart Image Logging:**
- Uses `logEpisodeWithImageControl()` function to optimize storage
- Logs Q-table heatmap images for:
  - First episode
  - Last episode  
  - N evenly spaced intermediate episodes (default: 10)
- Reduces storage costs while maintaining visualization quality

**Logged Metrics:**
- Episode metrics: steps, reward, duration, success rate
- Learning metrics: epsilon decay, consecutive optimal episodes
- Performance metrics: best path length, learning achievement status
- Q-table visualizations (when enabled)

**Configuration:**
- `USE_WANDB`: Enable/disable wandb logging (default: False)
- `N_IMAGE_EPISODES`: Number of intermediate episodes to log with images (default: 10)
- `SHOW_PLOTS`: Enable/disable matplotlib visualizations (affects wandb image logging)

### Running with wandb

```bash
# Enable wandb in main.py or main_nn.py first (set USE_WANDB = True)
uv run python main.py        # Tabular Q-learning
uv run python main_nn.py     # Deep Q-Network
```

**Note:** The local terminal visualizations using Rich are comprehensive and may be more useful for understanding the learning process than the wandb dashboard. wandb is primarily useful for:
- Long-term experiment tracking
- Comparing different hyperparameter configurations
- Sharing results with team members
- Advanced visualization and analysis
