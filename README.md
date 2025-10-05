# RL Gridworld

This is a Reinforcement Learning implementation of Q-Learning in both 1D and 2D gridworld environments. Why you ask? Well, the reason is to learn.

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
- **Scalable** to large state spaces
- **Experience replay** for stable learning
- **Best for**: Large grids (≥ 50x50), advanced RL techniques

For detailed comparison and usage guide, see [NEURAL_NETWORK_GUIDE.md](NEURAL_NETWORK_GUIDE.md).

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

You can modify the following parameters in `main.py`:

### Environment Parameters

- `grid_size_x`: Width of the 2D grid (default: 20)
- `grid_size_y`: Height of the 2D grid (default: 5)
- `start_pos`: Starting position (default: (0, 0))
- `goal_pos`: Goal position (default: (grid_size_x-1, grid_size_y-1))

### Training Parameters

- `num_episodes`: Number of training episodes (default: 500)
- `learning_rate`: Learning rate for Q-value updates (default: 0.1)
- `discount_factor`: Discount factor for future rewards (default: 0.99)
- `epsilon`: Initial exploration rate (default: 1.0)
- `epsilon_decay`: Decay rate for exploration (default: 0.99)
- `epsilon_min`: Minimum exploration rate (default: 0.01)

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
