import numpy as np

class GridWorld2D:
    def __init__(self, grid_size=5, start_pos=(0, 0), end_pos=(4, 4)):
        """
        Initialize a 2D gridworld environment.
        
        Args:
            grid_size (int): Size of the grid (grid_size x grid_size). Default is 5.
            start_pos (tuple): Starting position coordinates (x, y). Default is (0, 0).
            end_pos (tuple): Goal position coordinates (x, y). Default is (4, 4).
        """
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.current_pos = start_pos
        
        # Define possible actions
        self.actions = ['up', 'down', 'left', 'right']
        
        # Validate positions
        self._validate_position(start_pos)
        self._validate_position(end_pos)
        
    def _validate_position(self, pos):
        """
        Validate if a position is within the grid boundaries.
        
        Args:
            pos (tuple): Position coordinates (x, y)
            
        Raises:
            ValueError: If position is outside grid boundaries
        """
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ValueError(f"Position {pos} is outside grid boundaries (0 to {self.grid_size-1})")
    
    def reset(self):
        """
        Reset the environment to the starting position.
        
        Returns:
            tuple: Current position after reset
        """
        self.current_pos = self.start_pos
        return self.current_pos
    
    def get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            tuple: Current position
        """
        return self.current_pos
    
    def is_terminal(self):
        """
        Check if the current position is the goal state.
        
        Returns:
            bool: True if current position is the goal state, False otherwise
        """
        return self.current_pos == self.end_pos

    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (str): One of ['up', 'down', 'left', 'right']
            
        Returns:
            tuple: (next_state, reward, done)
                - next_state: tuple of (x, y) coordinates
                - reward: float, reward for the action
                - done: bool, whether episode is finished
        """
        if action not in self.actions:
            raise ValueError(f"Unknown action: {action}. Must be one of {self.actions}")
        
        x, y = self.current_pos
        
        # Calculate next position based on action
        if action == 'up':
            next_pos = (x, y - 1)
        elif action == 'down':
            next_pos = (x, y + 1)
        elif action == 'left':
            next_pos = (x - 1, y)
        elif action == 'right':
            next_pos = (x + 1, y)
            
        # Check if next position is valid
        try:
            self._validate_position(next_pos)
            self.current_pos = next_pos
        except ValueError:
            # If invalid, stay in current position
            next_pos = self.current_pos
        
        # Determine reward
        if next_pos == self.end_pos:
            reward = 1.0  # reached goal
        else:
            reward = -0.01  # small negative reward for each step to encourage efficiency
        
        # Check if episode is done
        done = (next_pos == self.end_pos)
        
        return next_pos, reward, done

    def render(self):
        """
        Visualize the current state of the gridworld.
        
        Returns:
            str: String representation of the grid
        """
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark start position
        x, y = self.start_pos
        grid[y][x] = 'S'
        
        # Mark end position
        x, y = self.end_pos
        grid[y][x] = 'G'
        
        # Mark current position
        x, y = self.current_pos
        if grid[y][x] not in ['S', 'G']:  # Don't overwrite start or goal
            grid[y][x] = 'A'
        
        # Convert grid to string representation
        grid_str = '\n'.join([' '.join(row) for row in grid])
        return grid_str