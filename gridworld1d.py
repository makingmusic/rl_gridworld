import numpy as np

# 1D Gridworld Environment
class GridWorld1D:
    def __init__(self, size=5, start_state=0, goal_state=None):
        """
        Initialize the 1D Gridworld.
        :param size: Number of cells in the grid.
        :param start_state: The starting cell index for each episode.
        :param goal_state: The terminal (goal) cell index. Defaults to last cell (size-1).
        """
        self.size = size
        self.start_state = start_state
        self.goal_state = goal_state if goal_state is not None else size - 1
        self.state = start_state  # current state of the environment
        
    def reset(self):
        """Reset the environment to start a new episode. Returns the start state."""
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        """
        Take an action in the environment.
        :param action: An action ('left' or 'right').
        :return: tuple (next_state, reward, done)
        """
        if action == "left":
            # move left (decrease state index)
            next_state = self.state - 1
        elif action == "right":
            # move right (increase state index)
            next_state = self.state + 1
        else:
            raise ValueError(f"Unknown action: {action}")
        
        # Enforce boundaries: if out of bounds, stay in same state
        if next_state < 0 or next_state >= self.size:
            next_state = self.state  # no movement if action is invalid (boundary)
        
        # Determine reward
        if next_state == self.goal_state:
            reward = 1.0  # reached goal
        else:
            #reward = 0.0  # no reward for staying in the same state
            reward = -0.1  # small negative reward for each step to encourage efficiency
        
        # Check if episode is done
        done = (next_state == self.goal_state)
        
        # Update the state of the environment
        self.state = next_state
        
        return next_state, reward, done
