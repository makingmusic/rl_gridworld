import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DQN(nn.Module):
    """Deep Q-Network model."""

    def __init__(self, state_size=2, action_size=4, hidden_size=128):
        """
        Initialize the DQN.
        :param state_size: Number of state features (x, y coordinates)
        :param action_size: Number of possible actions
        :param hidden_size: Size of hidden layers
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ExperienceReplay:
    """Experience replay buffer for storing and sampling experiences."""

    def __init__(self, buffer_size=10000):
        """
        Initialize the replay buffer.
        :param buffer_size: Maximum size of the buffer
        """
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent for 2D gridworld with same interface as tabular agent."""

    def __init__(
        self,
        actions,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        exploration_strategy="epsilon_greedy",
        grid_size_x=10,
        grid_size_y=10,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
        hidden_size=128,
        device=None,  # Add device argument
    ):
        """
        Initialize the DQN agent.
        :param actions: List of possible actions (e.g., ['up', 'down', 'left', 'right'])
        :param learning_rate: Learning rate for neural network
        :param discount_factor: Gamma, discount factor for future rewards
        :param epsilon: Initial epsilon for epsilon-greedy exploration
        :param epsilon_decay: Multiplicative factor to decay epsilon each episode
        :param epsilon_min: Minimum epsilon value for exploration
        :param exploration_strategy: Exploration strategy (only epsilon_greedy supported for now)
        :param grid_size_x: Width of the grid (for state normalization)
        :param grid_size_y: Height of the grid (for state normalization)
        :param buffer_size: Size of experience replay buffer
        :param batch_size: Batch size for training
        :param target_update_freq: Frequency to update target network
        :param hidden_size: Size of hidden layers in neural network
        :param device: torch.device to use for model/tensors (MPS, CUDA, or CPU)
        """
        self.actions = actions
        self.action_to_idx = {action: idx for idx, action in enumerate(actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(actions)}
        self.num_actions = len(actions)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exploration_strategy = exploration_strategy
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_step = 0

        # Set device
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_network = DQN(
            state_size=2, action_size=self.num_actions, hidden_size=hidden_size
        ).to(self.device)
        self.target_network = DQN(
            state_size=2, action_size=self.num_actions, hidden_size=hidden_size
        ).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize target network with same weights as main network
        self.update_target_network()

        # Initialize experience replay buffer
        self.memory = ExperienceReplay(buffer_size)

    def state_to_tensor(self, state):
        """Convert state tuple to normalized tensor."""
        x, y = state
        # Normalize coordinates to [0, 1] range
        normalized_state = [x / (self.grid_size_x - 1), y / (self.grid_size_y - 1)]
        return torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)

    def choose_action(self, state):
        """Choose an action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            # Exploration: choose random action
            return np.random.choice(self.actions)
        else:
            # Exploitation: choose best action according to Q-network
            state_tensor = self.state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            return self.idx_to_action[action_idx]

    def update_q_value(self, state, action, reward, next_state, done):
        """Store experience in replay buffer and train if enough samples available."""
        # Store experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Train if we have enough samples
        if len(self.memory) >= self.batch_size:
            self._train()

    def _train(self):
        """Train the Q-network using a batch from experience replay."""
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        state_batch = torch.FloatTensor(
            [
                [s[0] / (self.grid_size_x - 1), s[1] / (self.grid_size_y - 1)]
                for s in states
            ]
        ).to(self.device)
        action_batch = torch.LongTensor([self.action_to_idx[a] for a in actions]).to(
            self.device
        )
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(
            [
                [s[0] / (self.grid_size_x - 1), s[1] / (self.grid_size_y - 1)]
                for s in next_states
            ]
        ).to(self.device)
        done_batch = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        # Next Q-values from target network
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update training step counter
        self.training_step += 1

        # Update target network periodically
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay the exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def getQTable(self):
        """
        Generate a Q-table dictionary for compatibility with existing plotting code.
        This approximates the neural network's Q-values across the grid.
        """
        q_table = {}

        with torch.no_grad():
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    state = (x, y)
                    state_tensor = self.state_to_tensor(state)
                    q_values = self.q_network(state_tensor).squeeze().cpu().numpy()

                    q_table[state] = {
                        action: float(q_values[self.action_to_idx[action]])
                        for action in self.actions
                    }

        return q_table

    def getQTableAsPolicyArrows(self):
        """
        Convert neural network policy to arrow format for visualization.
        """
        q_table_policyarrows = {}

        arrows = {"up": 1, "down": 2, "left": 3, "right": 4}

        with torch.no_grad():
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    state = (x, y)
                    state_tensor = self.state_to_tensor(state)
                    q_values = self.q_network(state_tensor).squeeze().cpu().numpy()

                    # Find best action
                    best_action_idx = np.argmax(q_values)
                    best_action = self.idx_to_action[best_action_idx]

                    # Check if all values are essentially zero (untrained)
                    if np.max(np.abs(q_values)) < 1e-6:
                        q_table_policyarrows[state] = {"policyarrow": "0"}
                    else:
                        q_table_policyarrows[state] = {
                            "policyarrow": arrows[best_action]
                        }

        return q_table_policyarrows

    def print_q_table(self):
        """Print approximated Q-table for debugging."""
        print("\nApproximated Q-Table from Neural Network:")
        print("State (x,y) | Up    | Down  | Left  | Right")
        print("-" * 50)

        q_table = self.getQTable()
        sorted_states = sorted(q_table.keys())

        for state in sorted_states:
            actions = q_table[state]
            state_str = f"({state[0]},{state[1]})"
            up_val = f"{actions['up']:.3f}"
            down_val = f"{actions['down']:.3f}"
            left_val = f"{actions['left']:.3f}"
            right_val = f"{actions['right']:.3f}"

            print(
                f"{state_str:10} | {up_val:6} | {down_val:6} | {left_val:6} | {right_val:6}"
            )
