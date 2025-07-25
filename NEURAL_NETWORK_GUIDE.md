# Deep Q-Network (DQN) Implementation Guide

## Overview

This project now includes both **tabular Q-learning** and **Deep Q-Network (DQN)** implementations for the 2D gridworld environment. The DQN approach uses neural networks to approximate Q-values, making it scalable to much larger state spaces.

## Files Structure

### Tabular Q-Learning (Original)
- `main.py` - Main training script using tabular Q-learning
- `q_agent2D.py` - Tabular Q-learning agent implementation

### Deep Q-Network (New)
- `main_nn.py` - Main training script using DQN
- `q_agent_nn.py` - Neural network-based Q-learning agent

### Shared Components
- `gridworld2d.py` - Environment (unchanged)
- `plots.py` - Visualization utilities (works with both approaches)
- `logWandB.py` - Weights & Biases logging (unchanged)

## Key Differences

### Tabular Q-Learning vs Deep Q-Network

| Aspect | Tabular Q-Learning | Deep Q-Network (DQN) |
|--------|-------------------|---------------------|
| **Q-Value Storage** | Dictionary/Table | Neural Network |
| **Memory Usage** | O(states × actions) | O(network parameters) |
| **Scalability** | Limited to small state spaces | Scales to large/continuous spaces |
| **Learning** | Direct Q-value updates | Gradient-based optimization |
| **Convergence** | Guaranteed with proper conditions | Approximate, may be unstable |
| **Experience** | Uses current experience only | Experience replay buffer |
| **Training Speed** | Fast for small problems | Slower due to NN training |

## Neural Network Architecture

The DQN implementation uses a fully connected neural network:

```
Input Layer: 2 neurons (x, y coordinates, normalized)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 128 neurons (ReLU activation)
Hidden Layer 3: 128 neurons (ReLU activation)
Output Layer: 4 neurons (Q-values for each action)
```

## Key Components of DQN

### 1. Neural Network (`DQN` class)
- **Input**: Normalized (x, y) coordinates [0, 1]
- **Output**: Q-values for all 4 actions
- **Architecture**: 3 hidden layers with ReLU activation

### 2. Experience Replay Buffer (`ExperienceReplay` class)
- **Purpose**: Stores past experiences to break correlation between consecutive samples
- **Size**: 10,000 experiences by default
- **Sampling**: Random batch sampling for training

### 3. Target Network
- **Purpose**: Provides stable target Q-values during training
- **Update**: Copies main network weights every 100 training steps
- **Benefit**: Reduces oscillations and improves stability

### 4. Agent Interface (`DQNAgent` class)
Maintains the same interface as the tabular agent:
- `choose_action(state)` - Epsilon-greedy action selection
- `update_q_value(state, action, reward, next_state, done)` - Experience storage and training
- `getQTable()` - Approximates Q-table from neural network for visualization
- `getQTableAsPolicyArrows()` - Converts policy to arrow format

## Usage

### Running Tabular Q-Learning
```bash
python main.py
```

### Running Deep Q-Network
```bash
python main_nn.py
```

## Configuration Parameters

### DQN-Specific Parameters
```python
# Neural Network Parameters
learning_rate = 0.001          # Lower than tabular (0.1)
buffer_size = 10000           # Experience replay buffer size
batch_size = 32               # Training batch size
target_update_freq = 100      # Target network update frequency
hidden_size = 128             # Hidden layer size

# Training Parameters
num_episodes = 1000           # More episodes needed for NN
epsilon_decay = 0.995         # Slower decay for better exploration
```

### Key Differences in Parameters
- **Learning Rate**: DQN uses 0.001 vs tabular 0.1 (neural networks need smaller steps)
- **Episodes**: DQN typically needs more episodes (1000 vs 500)
- **Epsilon Decay**: Slower decay (0.995 vs 0.99) for better exploration

## Performance Characteristics

### Tabular Q-Learning
- **Pros**: 
  - Fast convergence for small grids
  - Guaranteed optimal policy with proper parameters
  - Simple to understand and debug
  - Exact Q-values
- **Cons**:
  - Memory grows exponentially with state space
  - Cannot generalize to unseen states
  - Limited to discrete, small state spaces

### Deep Q-Network
- **Pros**:
  - Scales to large state spaces
  - Generalizes to similar states
  - Memory efficient for large problems
  - Can handle continuous state spaces (with modifications)
- **Cons**:
  - Slower training due to neural network optimization
  - May not converge or find suboptimal policies
  - Hyperparameter sensitive
  - Requires more episodes to learn

## When to Use Each Approach

### Use Tabular Q-Learning When:
- State space is small (< 10,000 states)
- You need exact Q-values
- Training time is critical
- You want guaranteed convergence
- Problem is well-defined and discrete

### Use Deep Q-Network When:
- State space is large (> 10,000 states)
- Memory is limited
- You need generalization to similar states
- State space is continuous or high-dimensional
- You're planning to extend to more complex environments

## Visualization

Both implementations use the same visualization system:
- **Grid Display**: Shows current policy as arrows
- **Path Display**: Shows optimal path from start to goal
- **Progress Bars**: Real-time training progress
- **Plots**: Steps per episode and epsilon decay

The DQN agent approximates a Q-table for visualization by querying the neural network for all grid positions.

## Weights & Biases Integration

Both implementations support wandb logging with:
- Training metrics (steps, rewards, epsilon)
- Q-table heatmaps
- Episode statistics
- Neural network specific metrics (buffer size, training steps)

DQN adds additional metrics:
- `replay_buffer_size`: Current experience buffer size
- `training_step`: Number of neural network training steps
- `device`: GPU/CPU usage

## Tips for Best Results

### For Tabular Q-Learning:
1. Start with default parameters
2. Adjust epsilon decay based on grid size
3. Monitor Q-table convergence

### For Deep Q-Network:
1. **Warm-up Period**: Let the agent explore randomly for first 100-200 episodes
2. **Learning Rate**: Start with 0.001, adjust if training is unstable
3. **Buffer Size**: Ensure buffer is large enough (10x batch size minimum)
4. **Target Update**: More frequent updates (every 50-100 steps) for small problems
5. **Monitor Training**: Watch for Q-value explosion or collapse

## Troubleshooting

### Common DQN Issues:
1. **Q-values exploding**: Reduce learning rate or increase target update frequency
2. **No learning**: Increase exploration (higher epsilon, slower decay)
3. **Unstable performance**: Increase buffer size or batch size
4. **Memory issues**: Reduce buffer size or batch size

### Performance Comparison:
- DQN typically needs 2-5x more episodes than tabular for same performance
- DQN training is 3-10x slower per episode
- DQN memory usage is constant regardless of grid size
- Tabular memory grows quadratically with grid dimensions

## Extension Ideas

The DQN implementation is designed to be easily extensible:

1. **Double DQN**: Reduce overestimation bias
2. **Dueling DQN**: Separate value and advantage estimation
3. **Prioritized Experience Replay**: Sample important experiences more often
4. **Convolutional Networks**: For image-based states
5. **Larger Grids**: Test scalability with 50x50 or 100x100 grids
6. **Continuous Actions**: Extend to continuous action spaces

## Requirements

The neural network implementation adds PyTorch as a dependency:
```bash
pip install torch==2.1.0
```

All other dependencies remain the same. 