import numpy as np

# Q-learning Agent
class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1, exploration_strategy="epsilon_greedy", temperature=1.0, temperature_decay=0.99, temperature_min=0.1):
        """
        Initialize the Q-learning agent.
        :param actions: List of possible actions (e.g., ['left', 'right']).
        :param learning_rate: Alpha, the learning rate for Q-value updates.
        :param discount_factor: Gamma, the discount factor for future rewards.
        :param epsilon: Initial epsilon for epsilon-greedy exploration.
        :param epsilon_decay: Multiplicative factor to decay epsilon each episode.
        :param epsilon_min: Minimum epsilon value (floor) for exploration.
        :param exploration_strategy: Either "epsilon_greedy" or "softmax".
        :param temperature: Temperature parameter for softmax exploration.
        :param temperature_decay: Multiplicative factor to decay temperature each episode. softmax only.
        :param temperature_min: Minimum temperature value (floor) for softmax exploration.
        """
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exploration_strategy = exploration_strategy
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min
        # Initialize Q-table as an empty dict. Will add states as keys when encountered.
        self.q_table = {}  # format: {state: {action: value, ...}, ...}
    
    def softmax(self, action_values):
        """Compute softmax probabilities for action selection."""
        # Subtract max value for numerical stability
        exp_values = np.exp((action_values - np.max(action_values)) / self.temperature)
        return exp_values / np.sum(exp_values)
    
    def choose_action(self, state):
        """Choose an action based on the selected exploration strategy."""
        # Ensure the state exists in Q-table
        if state not in self.q_table:
            # Initialize with 0.0 for all possible actions
            self.q_table[state] = {a: 0.0 for a in self.actions}
        
        if self.exploration_strategy == "epsilon_greedy":
            # Epsilon-greedy exploration
            if np.random.rand() < self.epsilon:
                # Exploration: choose a random action
                return np.random.choice(self.actions)
            else:
                # Exploitation: choose the action with highest Q-value for this state
                action_values = self.q_table[state]
                best_action = max(action_values, key=action_values.get)
                return best_action
        else:  # softmax
            # Get Q-values for all actions in current state
            action_values = list(self.q_table[state].values())
            # Compute softmax probabilities
            probabilities = self.softmax(action_values)
            # Choose action based on computed probabilities
            return np.random.choice(self.actions, p=probabilities)
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update the Q-table entry for (state, action) using the Q-learning update rule."""
        # Ensure next_state is in Q-table (to allow grabbing max future Q). 
        # If terminal (done) next_state, we don't actually need to expand it, but doing no harm.
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}
        
        # Current Q-value for this state-action
        current_q = self.q_table[state][action]
        # Maximum Q-value for next state (next state's best action)
        max_future_q = max(self.q_table[next_state].values()) if not done else 0.0
        # Q-learning formula
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        # Update the Q-table
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Decay the exploration rate after each episode (to gradually favor exploitation)."""
        # Multiply epsilon by decay factor but not go below the minimum threshold
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # For softmax, we can also decay the temperature
        if self.exploration_strategy == "softmax":
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)  # Decay temperature using configurable parameters

    def getQTable(self):
        """Get the Q-table."""
        return self.q_table

    def print_q_table(self):
        """Print the Q-table."""
        for s, actions in self.q_table.items():
            qvalue = actions.values()
            qvaluelist = list(qvalue)
            qvaluelist_left = qvaluelist[0]
            qvaluelist_right = qvaluelist[1]
            qvaluelist_leftValue = round(float(qvaluelist_left), 4)
            qvaluelist_rightValue = round(float(qvaluelist_right), 4)
            print("State: ", s, ": Left: ", qvaluelist_leftValue, "Right: ", qvaluelist_rightValue)
            