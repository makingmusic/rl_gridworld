import numpy as np


class GridWorld2D:
    def __init__(
        self,
        grid_size_x=5,
        grid_size_y=5,
        start_pos=(0, 0),
        end_pos=(4, 4),
        max_steps_per_episode=None,
    ):
        """
        Initialize a 2D gridworld environment.

        Args:
            grid_size_x (int): Width of the grid. Default is 5.
            grid_size_y (int): Height of the grid. Default is 5.
            start_pos (tuple): Starting position coordinates (x, y). Default is (0, 0).
            end_pos (tuple): Goal position coordinates (x, y). Default is (4, 4).
            max_steps_per_episode (int | None): Maximum steps before episode terminates. If None, defaults to 4 * (grid_size_x + grid_size_y).
            Note: (0,0) is at the bottom left, with x increasing right and y increasing up.
        """
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.current_pos = start_pos
        self.steps_since_reset = 0
        # Default cap proportional to grid perimeter distance
        self.max_steps_per_episode = (
            max_steps_per_episode
            if max_steps_per_episode is not None
            else 4 * (self.grid_size_x + self.grid_size_y)
        )

        # Define possible actions
        self.actions = ["up", "down", "left", "right"]

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
        if not (0 <= x < self.grid_size_x and 0 <= y < self.grid_size_y):
            raise ValueError(
                f"Position {pos} is outside grid boundaries (x: 0 to {self.grid_size_x - 1}, y: 0 to {self.grid_size_y - 1})"
            )

    def reset(self):
        """
        Reset the environment to the starting position.

        Returns:
            tuple: Current position after reset
        """
        self.current_pos = self.start_pos
        self.steps_since_reset = 0
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
        if action == "up":
            next_pos = (x, y + 1)  # y increases up
        elif action == "down":
            next_pos = (x, y - 1)  # y decreases down
        elif action == "left":
            next_pos = (x - 1, y)
        elif action == "right":
            next_pos = (x + 1, y)

        # Check if next position is valid
        try:
            self._validate_position(next_pos)
            self.current_pos = next_pos
        except ValueError:
            # If invalid, stay in current position
            next_pos = self.current_pos

        # Increment step count after resolving movement
        self.steps_since_reset += 1

        # Determine reward with grid-size-aware scaling
        if next_pos == self.end_pos:
            # Scale goal reward with grid area to ensure it outweighs step penalties
            # Goal reward should be at least 2x the maximum possible step penalty
            max_possible_steps = self.max_steps_per_episode
            max_step_penalty = max_possible_steps * 0.01  # Base step penalty
            reward = max(10.0, 2.0 * max_step_penalty)  # Ensure goal reward is substantial
        else:
            # Scale step penalty inversely with grid size to prevent over-penalization
            # For larger grids, use smaller per-step penalties
            base_penalty = 0.01
            grid_area = self.grid_size_x * self.grid_size_y
            # Scale penalty: smaller grids get higher penalties, larger grids get lower penalties
            scaled_penalty = base_penalty * (100.0 / grid_area) ** 0.5  # Square root scaling
            reward = -scaled_penalty

        # Check if episode is done by reaching goal or exceeding max steps
        if next_pos == self.end_pos:
            done = True
        elif self.steps_since_reset >= self.max_steps_per_episode:
            done = True
            # Apply timeout penalty proportional to grid size
            timeout_penalty = max(0.1, 0.01 * (self.grid_size_x + self.grid_size_y))
            reward = -timeout_penalty
        else:
            done = False

        return next_pos, reward, done

    def render(self):
        """
        Visualize the current state of the gridworld.
        The grid is displayed with (0,0) at the bottom left.

        Returns:
            str: String representation of the grid
        """
        grid = [["." for _ in range(self.grid_size_x)] for _ in range(self.grid_size_y)]

        # Mark start position
        x, y = self.start_pos
        grid[y][x] = "S"  # y is already in the correct orientation

        # Mark end position
        x, y = self.end_pos
        grid[y][x] = "G"  # y is already in the correct orientation

        # Mark current position
        x, y = self.current_pos
        if grid[y][x] not in ["S", "G"]:  # Don't overwrite start or goal
            grid[y][x] = "A"

        # Convert grid to string representation, reversing the rows to show (0,0) at bottom
        grid_str = "\n".join([" ".join(row) for row in reversed(grid)])
        return grid_str
