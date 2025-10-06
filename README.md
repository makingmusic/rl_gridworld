# RL Gridworld

This is a Reinforcement Learning implementation of Q-Learning in both 1D and 2D gridworld environments. Why you ask? Well, the reason is to learn.

Note2Self: 100x100 grid trained in about 2 hours on an M3 Mac / 24 Gb. Need to do more benchmarking / profiling. So much time is wasted in just measuring while training. Measuring ought to occur on a separate machine i I really want to do this properly.

## Quick Start

To get started immediately, simply run this command in your terminal:

```bash
curl -sSL https://raw.githubusercontent.com/makingmusic/rl_gridworld/main/run.sh | bash
```

This will:

1. Set up a Python virtual environment
2. Install all required dependencies
3. Provide you with the command to run the 1D/2D grid world example

After installation, you can run:

- `python main.py` for tabular Q-learning in the 2D grid world
- `python main_nn.py` for Deep Q-Network (DQN) in the 2D grid world

## The Environment

### 2D Gridworld

A 1D/2D world where the agent can move up, down, left, or right. The goal is to reach the top-right corner. If it is setup with just one row, then it can simulate a 1D world.

Rules:

- Agent can move in four directions: up, down, left, right
- Cannot move outside the grid boundaries
- Goal state is at the top-right corner (grid_size_x-1, grid_size_y-1)
- Default grid size is 20x5

## The Agent

The agent implements Q-Learning with the following features:

1. **Epsilon-greedy Exploration Strategy**:

   - Starts with high exploration (epsilon = 1.0)
   - Gradually reduces exploration through epsilon decay to 0.01

2. **Q-Value Updates**:
   - Learning rate: 0.1
   - Discount factor: 0.99

## Implementation Approaches

This project offers two different Q-learning implementations:

### 1. Tabular Q-Learning (`main.py`)
- **Traditional approach** using lookup tables
- **Fast convergence** for small state spaces
- **Exact Q-values** stored in memory
- **Best for**: Small grids (≤ 20x20), learning RL fundamentals

### 2. Deep Q-Network - DQN (`main_nn.py`)
- **Neural network** approximates Q-values
- **Adaptive sizing** automatically scales network capacity based on grid size
- **Experience replay** for stable learning
- **Target network** for stable Q-value estimation
- **Best for**: Large grids (≥ 50x50), advanced RL techniques

For detailed comparison and usage guide, see [NEURAL_NETWORK_GUIDE.md](NEURAL_NETWORK_GUIDE.md).

## Reward Structure & Grid Size Scaling

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

**Implementation Location**: The reward function is defined in `gridworld2d.py` in the `step()` method (lines 128-155).

## Adaptive Neural Network Sizing

### The Problem: Fixed Network Size Limitations

The original DQN implementation used a fixed neural network architecture regardless of grid size:

**Previous Fixed Architecture:**
- **Hidden layers**: 3 layers of 128 neurons each
- **Total parameters**: ~33,000 parameters
- **Buffer size**: 10,000 experiences (fixed)
- **Batch size**: 64 (fixed)

**Why This Failed for Large Grids:**
- **50×50 grid**: 2,500 states → 13.3 parameters per state (adequate)
- **100×100 grid**: 10,000 states → 3.3 parameters per state (insufficient!)
- **200×200 grid**: 40,000 states → 0.8 parameters per state (severely inadequate!)

The fixed network became increasingly under-capacity as grid size increased, leading to poor learning performance and convergence issues.

### The Solution: Adaptive Neural Network Sizing

**New Adaptive Architecture** automatically scales all parameters based on grid dimensions:

#### 1. **Adaptive Hidden Layer Size**
```python
def compute_optimal_nn_size(grid_x, grid_y, min_hidden=128, max_hidden=1024):
    total_states = grid_x * grid_y
    # Target 15 parameters per state for optimal learning capacity
    optimal_hidden = int(np.sqrt(7.5 * total_states))
    # Apply bounds and round to power of 2 for efficiency
    optimal_hidden = max(min_hidden, min(optimal_hidden, max_hidden))
    optimal_hidden = 2 ** int(np.log2(optimal_hidden) + 0.5)
    return optimal_hidden
```

#### 2. **Adaptive Buffer Size**
```python
def compute_adaptive_buffer_size(grid_x, grid_y, base_size=10000):
    total_states = grid_x * grid_y
    # Scale with square root of grid area for diverse experiences
    adaptive_size = min(base_size * (total_states / 2500) ** 0.5, 100000)
    return int(adaptive_size)
```

#### 3. **Adaptive Batch Size**
```python
def compute_adaptive_batch_size(grid_x, grid_y, base_size=64):
    total_states = grid_x * grid_y
    # Scale with grid area to power 0.25 for stable training
    adaptive_size = min(base_size * (total_states / 2500) ** 0.25, 256)
    return int(adaptive_size)
```

### Results: Optimal Capacity Across All Grid Sizes

| Grid Size | States | Hidden Size | Buffer Size | Batch Size | NN Parameters | Params/State |
|-----------|--------|-------------|-------------|------------|---------------|--------------|
| 10×10     | 100    | 128         | 2,000       | 28         | 33,280        | 332.8        |
| 25×25     | 625    | 128         | 5,000       | 45         | 33,280        | 53.2         |
| 50×50     | 2,500  | 128         | 10,000      | 64         | 33,280        | 13.3         |
| 100×100   | 10,000 | 256         | 20,000      | 90         | 132,096       | 13.2         |
| 200×200   | 40,000 | 512         | 40,000      | 128        | 526,336       | 13.2         |

### Key Benefits

- ✅ **Automatic scaling**: No manual tuning needed for different grid sizes
- ✅ **Optimal capacity**: Maintains ~13-15 parameters per state for consistent learning
- ✅ **Computational efficiency**: Uses powers of 2 for GPU optimization
- ✅ **Memory management**: Reasonable bounds prevent excessive memory usage
- ✅ **Better convergence**: Larger grids get appropriately sized networks
- ✅ **Future-proof**: Works optimally for any grid size you choose

### Algorithm Details

**Target Capacity**: 15 parameters per state (research-backed optimal ratio)
**Scaling Formula**: `hidden_size ≈ sqrt(7.5 × total_states)`
**Bounds**: 128 ≤ hidden_size ≤ 1024 (powers of 2 for efficiency)
**Buffer Scaling**: `buffer_size ∝ sqrt(grid_area)` for diverse experiences
**Batch Scaling**: `batch_size ∝ grid_area^0.25` for stable training

**Implementation Location**: The adaptive sizing functions are defined in `main_nn.py` (lines 53-107) and automatically applied during agent initialization.

## Weights & Biases (wandb) Integration

[Weights & Biases (wandb)](https://wandb.ai/). By default, wandb logging is enabled.

### Setting up wandb

**Step 1: Install wandb**

```bash
pip install wandb
```

**Step 2: Create a wandb account and login**

1. Go to [wandb.ai](https://wandb.ai/) and create a free account
2. Login to wandb in your terminal:

```bash
wandb login
```

3. Find your API key at [wandb.ai/authorize](https://wandb.ai/authorize)

**Step 3: Configure the project**

- By default, wandb logging is enabled (`USE_WANDB = True` in `main.py`).
- If you do not wish to use wandb, set `USE_WANDB = False` in `main.py`.
- Projects are logged separately:
  - Tabular Q-learning: `rl-gridworld-qlearning`
  - Deep Q-Network: `rl-gridworld-dqn`

**Note:** If you prefer not to use wandb, simply set `USE_WANDB = False` in `main.py` and the program will run without any online logging. The local terminal visualizations using Rich are actually quite comprehensive and may be more useful for understanding the learning process than the wandb dashboard.

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

### Display Parameters

- `sleep_time`: Time to pause between episodes (default: 0)

### Episode Step Cap

To prevent episodes from wandering too long under high exploration, the environment enforces a `max_steps_per_episode` cap:

- In the environment (`gridworld2d.py`): `GridWorld2D` accepts `max_steps_per_episode` and terminates the episode with a small timeout penalty when exceeded.
- In the tabular script (`main.py`): configure `max_steps_per_episode` in the Grid Configuration section and it is passed to `GridWorld2D`.
- In the DQN script (`main_nn.py`): `max_steps_per_episode` is computed by `compute_max_steps(grid_x, grid_y, epsilon)` which scales with Manhattan distance and grid area, with an optional adaptive mode based on recent episode lengths.

Display note: The steps summary shows entries like `42(M)` in "Steps in last 10 episodes:" when an episode ends by hitting the max step cap.

## Visualization Features

The project includes several visualization features to help understand the learning process:

## Installation

1. Clone the repository:

```bash
git clone https://github.com/makingmusic/rl_gridworld.git
cd rl_gridworld
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script to start the training:

```bash
python main.py
```

When you run the script for the first time with wandb enabled, you may be prompted to log in or create a wandb account. You can skip this if you do not wish to log online, or disable wandb logging in `main.py`.

## Future Improvements

- starting at different points in the state machine (done)
- having the goal be another point than the extreme state on the right. (done)
- allow "jumps", which may be X number of steps that can be taken together. Would be fun to see
  how learning improves (or becomes worse) by higher values of X.
- Add obstacles or forbidden states so the only way to reach the end would be jump over them.
- Changing the tradeoffs between exploitation vs exploration. There are many algorithms available
  include the infamous softmax that are used in modern LLMs too. I want to get there.

Potential areas for future development:

1. **Advanced Exploration Strategies**:

   - Upper Confidence Bound (UCB)
   - Thompson Sampling
   - Boltzmann Exploration

2. **Environment Enhancements**:

   - Multiple goal states
   - Obstacles and forbidden states
   - Stochastic transitions
   - Variable step sizes

3. **Learning Algorithm Extensions**:

   - Double Q-Learning
   - Prioritized Experience Replay
   - Dueling Network Architectures

4. **Visualization Improvements**:
   - Interactive plots (done, using wandb)
   - 3D value function visualization
   - Policy heatmaps

## GPT Says I should try the following:

1. Implement Multiple RL Algorithms
   • Value Iteration & Policy Iteration: Understand the differences between model-based and model-free approaches.
   • Monte Carlo Methods: Explore how sampling can be used for policy evaluation and improvement.
   • Temporal Difference Learning: Implement SARSA and Q-learning to compare on-policy and off-policy methods.

2. Introduce Function Approximation
   • Linear Function Approximation: Replace tabular methods with linear approximators to handle larger state spaces.
   • Neural Networks: Begin with simple feedforward networks before moving to more complex architectures.

3. Experiment with Exploration Strategies
   • Epsilon-Greedy: Analyze how varying epsilon affects learning.
   • Softmax Action Selection: Implement and compare with epsilon-greedy.
   • Upper Confidence Bound (UCB): Explore how optimism in the face of uncertainty can drive exploration.

4. Incorporate Stochasticity
   • Action Noise: Introduce randomness in action outcomes to simulate real-world unpredictability.
   • Reward Noise: Add variability to rewards to study robustness.

5. Visualize Learning Progress
   • Heatmaps of State-Value Functions: Visualize how the agent's understanding of the environment evolves.
   • Policy Arrows: Display the agent's preferred action in each state.

## License

I don't even understand how licensing works across MIT and whatnot. I wrote this with a large amount of help from ChatGPT and if you find it useful, please use it in any way you feel like without any obligations to me. And I welcome your feedback if any. Personally I learnt so much that I could not have kept it hidden in my laptop so here you go !
