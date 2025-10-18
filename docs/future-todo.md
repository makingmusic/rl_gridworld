## Memory-Augmented Neural Networks for Wall Navigation

### The Challenge: Learning to Navigate Obstacles

The current DQN implementation has a fundamental limitation: **it has no memory of past states**. This severely limits its ability to navigate complex environments with walls/obstacles, where the agent needs to:

- Remember where it has been before
- Avoid revisiting dead ends
- Learn efficient backtracking strategies
- Navigate new wall configurations it has never seen during training

### The Memory Problem

**Current State Representation:**
```python
state = (x, y)  # Just current position - no memory!
```

**What the NN Lacks:**
- Knowledge of previously visited positions
- Memory of where walls were encountered
- Awareness of which paths have already been tried
- Understanding of the overall environment structure

**Result:** The agent repeatedly tries the same failed paths, leading to inefficient exploration and poor navigation performance.

### Solution: Memory-Augmented Architectures

To enable effective wall navigation, the neural network needs memory. Here are the key approaches to implement:

#### 1. **LSTM-Based DQN (Recommended)**

**Concept:** Use Long Short-Term Memory networks to maintain hidden states that remember previous observations.

**Architecture:**
```python
class LSTM_DQN(nn.Module):
    def __init__(self, state_size=2, action_size=4, hidden_size=128, sequence_length=10):
        super().__init__()
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)
        self.sequence_length = sequence_length
    
    def forward(self, state_sequence):
        # state_sequence: (batch_size, sequence_length, state_size)
        lstm_out, _ = self.lstm(state_sequence)
        # Use the last output for action selection
        q_values = self.fc(lstm_out[:, -1, :])
        return q_values
```

**How It Works:**
- Input: Sequence of recent states `[(x1,y1), (x2,y2), ..., (x_n, y_n)]`
- LSTM: Processes the sequence, maintaining memory of past observations
- Output: Q-values informed by exploration history

**Benefits:**
- Remembers walls encountered earlier in the episode
- Avoids revisiting dead ends
- Learns efficient exploration strategies
- Develops backtracking behaviors when needed

#### 2. **Attention-Based Memory**

**Concept:** Use multi-head attention mechanisms to focus on relevant past states.

**Architecture:**
```python
class Attention_DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        # Can attend to previous states in the episode
```

**Benefits:**
- Selective memory - focuses on important past observations
- Can handle variable-length sequences
- More interpretable than LSTM (attention weights show what the agent is "thinking about")

#### 3. **External Memory Systems**

**Concept:** Maintain explicit memory buffers outside the neural network.

**Implementation:**
```python
class DQNWithExternalMemory:
    def __init__(self):
        self.visited_positions = set()
        self.wall_positions = set()
        self.episode_memory = []
    
    def update_memory(self, state, action, reward, next_state, done):
        self.visited_positions.add(state)
        if reward < -0.5:  # Assuming wall hits give negative reward
            self.wall_positions.add(next_state)
        self.episode_memory.append((state, action, reward, next_state))
```

**Benefits:**
- Explicit control over what is remembered
- Can implement sophisticated memory management
- Easier to debug and understand

#### 4. **Hybrid Approach**

**Concept:** Combine standard DQN with external memory for episode-level planning.

**Implementation:**
```python
class HybridDQN:
    def __init__(self):
        self.dqn = DQN()  # Standard DQN for immediate decisions
        self.memory = EpisodeMemory()  # External memory for episode-level planning
    
    def choose_action(self, state):
        q_values = self.dqn(state)
        
        # Penalize actions that lead to recently visited states
        for action in self.actions:
            next_state = self.get_next_state(state, action)
            if next_state in self.recently_visited:
                q_values[action] -= 0.1  # Small penalty
        
        return self.select_action(q_values)
```

### Implementation Roadmap

#### Phase 1: Add Walls to Environment
1. **Modify `GridWorld2D`** to support wall placement
2. **Implement maze generation** algorithms that guarantee connectivity
3. **Add wall collision detection** and appropriate rewards
4. **Create diverse wall configurations** for training

#### Phase 2: Implement LSTM-Based DQN
1. **Replace feedforward DQN** with LSTM-based architecture
2. **Modify state representation** to use sequences of recent states
3. **Update training loop** to handle sequential data
4. **Implement proper sequence padding** and batching

#### Phase 3: Training Strategy
1. **Curriculum learning**: Start with simple obstacles, increase complexity
2. **Diverse obstacle patterns**: Train on many different wall configurations
3. **Generalization testing**: Evaluate on completely unseen obstacle layouts
4. **Success metrics**: Path optimality, success rate, exploration efficiency

#### Phase 4: Advanced Memory Architectures
1. **Experiment with attention mechanisms** for selective memory
2. **Implement external memory systems** for explicit episode planning
3. **Compare different memory approaches** on navigation tasks
4. **Optimize for real-time performance** in larger environments

### Expected Benefits

With memory-augmented neural networks, the agent will be able to:

- **Navigate complex mazes** with walls and obstacles
- **Generalize to new environments** it has never seen during training
- **Learn efficient exploration strategies** that avoid redundant paths
- **Develop sophisticated backtracking behaviors** when dead ends are encountered
- **Scale to larger, more complex environments** with realistic navigation challenges

### Key Research Questions

1. **How much memory is needed?** What's the optimal sequence length for LSTM?
2. **What's the best memory architecture?** LSTM vs. Attention vs. External memory?
3. **How to balance exploration vs. exploitation** in memory-augmented systems?
4. **Can the agent learn to build mental maps** of the environment?
5. **How does memory capacity scale** with environment complexity?

This represents a significant step from simple pathfinding to true spatial reasoning and navigation - a fundamental capability for real-world AI systems.

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