# RL Gridworld

A Reinforcement Learning implementation of Q-Learning in a 1D gridworld environment. This project demonstrates how an agent learns to navigate from a starting position to a goal state using Q-Learning algorithm.

## Features

- 1D Gridworld environment with configurable size
- Q-Learning agent with epsilon-greedy exploration strategy
- Real-time visualization of training progress using Rich
- Interactive Q-table display
- Progress tracking with multiple metrics
- Visualization of learning progress through matplotlib plots

## Requirements

- Python 3.x
- Required packages (install using `pip install -r requirements.txt`):
  - numpy
  - matplotlib
  - rich

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

The program will:

1. Initialize a 1D gridworld environment
2. Create a Q-Learning agent
3. Train the agent for the specified number of episodes
4. Display real-time progress using Rich
5. Show final Q-values and learning curves

## Configuration

You can modify the following parameters in `main.py`:

- `num_episodes`: Number of training episodes (default: 1000)
- `grid1DSize`: Size of the 1D grid (default: 100)
- `startState`: Starting position (default: 0)
- `goalState`: Goal position (default: grid1DSize - 1)
- `sleep_time`: Time to pause between episodes (default: 0)

## Visualization

The program provides several visualizations:

- Real-time Q-table display
- Training progress bar
- State progress tracking
- Learning curves showing:
  - Steps per episode
  - Q-values for selected states

## License

This project is open source and available under the MIT License.
