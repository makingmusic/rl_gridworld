import numpy as np
import random
from collections import deque


class GridWorld2D:
    def __init__(
        self,
        grid_size_x=5,
        grid_size_y=5,
        start_pos=(0, 0),
        end_pos=(4, 4),
        max_steps_per_episode=None,
        obstacles=None,
    ):
        """
        Initialize a 2D gridworld environment.

        Args:
            grid_size_x (int): Width of the grid. Default is 5.
            grid_size_y (int): Height of the grid. Default is 5.
            start_pos (tuple): Starting position coordinates (x, y). Default is (0, 0).
            end_pos (tuple): Goal position coordinates (x, y). Default is (4, 4).
            max_steps_per_episode (int | None): Maximum steps before episode terminates. If None, defaults to 4 * (grid_size_x + grid_size_y).
            obstacles (set | None): Set of obstacle positions (x, y). Default is None (no obstacles).
            Note: (0,0) is at the bottom left, with x increasing right and y increasing up.
        """
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.current_pos = start_pos
        self.steps_since_reset = 0
        self.obstacles = obstacles if obstacles is not None else set()
        
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
        
        # Validate that start and goal positions are not obstacles
        if start_pos in self.obstacles:
            raise ValueError(f"Start position {start_pos} cannot be an obstacle")
        if end_pos in self.obstacles:
            raise ValueError(f"Goal position {end_pos} cannot be an obstacle")

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

    def _is_valid_position(self, pos):
        """
        Check if a position is valid (within bounds and not an obstacle).

        Args:
            pos (tuple): Position coordinates (x, y)

        Returns:
            bool: True if position is valid, False otherwise
        """
        x, y = pos
        return (0 <= x < self.grid_size_x and 
                0 <= y < self.grid_size_y and 
                pos not in self.obstacles)

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

        # Check if next position is valid (within bounds and not an obstacle)
        if self._is_valid_position(next_pos):
            self.current_pos = next_pos
        else:
            # If invalid (out of bounds or obstacle), stay in current position
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
            # Check if agent hit a wall/obstacle (stayed in same position)
            if next_pos == self.current_pos and action in ["up", "down", "left", "right"]:
                # Calculate next position based on action to check if it would be invalid
                x, y = self.current_pos
                if action == "up":
                    attempted_pos = (x, y + 1)
                elif action == "down":
                    attempted_pos = (x, y - 1)
                elif action == "left":
                    attempted_pos = (x - 1, y)
                elif action == "right":
                    attempted_pos = (x + 1, y)
                
                # If attempted position is invalid, apply bump penalty
                if not self._is_valid_position(attempted_pos):
                    # Small bump penalty for hitting walls
                    base_penalty = 0.01
                    grid_area = self.grid_size_x * self.grid_size_y
                    scaled_penalty = base_penalty * (100.0 / grid_area) ** 0.5
                    reward = -scaled_penalty * 2  # Double penalty for hitting walls
                else:
                    # Normal step penalty
                    base_penalty = 0.01
                    grid_area = self.grid_size_x * self.grid_size_y
                    scaled_penalty = base_penalty * (100.0 / grid_area) ** 0.5
                    reward = -scaled_penalty
            else:
                # Normal step penalty
                base_penalty = 0.01
                grid_area = self.grid_size_x * self.grid_size_y
                scaled_penalty = base_penalty * (100.0 / grid_area) ** 0.5
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

        # Mark obstacles
        for x, y in self.obstacles:
            grid[y][x] = "#"  # Use # for obstacles

        # Mark start position
        x, y = self.start_pos
        grid[y][x] = "S"  # y is already in the correct orientation

        # Mark end position
        x, y = self.end_pos
        grid[y][x] = "G"  # y is already in the correct orientation

        # Mark current position
        x, y = self.current_pos
        if grid[y][x] not in ["S", "G", "#"]:  # Don't overwrite start, goal, or obstacles
            grid[y][x] = "A"

        # Convert grid to string representation, reversing the rows to show (0,0) at bottom
        grid_str = "\n".join([" ".join(row) for row in reversed(grid)])
        return grid_str


def a_star_pathfinding(grid_size_x, grid_size_y, start_pos, goal_pos, obstacles):
    """
    A* pathfinding algorithm to find optimal path from start to goal.
    
    Args:
        grid_size_x (int): Grid width
        grid_size_y (int): Grid height
        start_pos (tuple): Starting position (x, y)
        goal_pos (tuple): Goal position (x, y)
        obstacles (set): Set of obstacle positions (x, y)
    
    Returns:
        tuple: (path_length, path) where path_length is the number of steps,
               or (None, None) if no path exists
    """
    def heuristic(pos):
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
    
    def get_neighbors(pos):
        """Get valid neighboring positions."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # up, down, right, left
            new_pos = (x + dx, y + dy)
            if (0 <= new_pos[0] < grid_size_x and 
                0 <= new_pos[1] < grid_size_y and 
                new_pos not in obstacles):
                neighbors.append(new_pos)
        return neighbors
    
    # A* algorithm
    open_set = [(0, start_pos)]  # (f_score, position)
    came_from = {}
    g_score = {start_pos: 0}
    f_score = {start_pos: heuristic(start_pos)}
    
    while open_set:
        open_set.sort()  # Sort by f_score
        current_f, current = open_set.pop(0)
        
        if current == goal_pos:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_pos)
            path.reverse()
            return len(path) - 1, path  # Return path length and path
        
        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor)
                
                if not any(pos == neighbor for _, pos in open_set):
                    open_set.append((f_score[neighbor], neighbor))
    
    return None, None  # No path found


def is_solvable(grid_size_x, grid_size_y, start_pos, goal_pos, obstacles):
    """
    Check if a grid configuration is solvable using A*.
    
    Args:
        grid_size_x (int): Grid width
        grid_size_y (int): Grid height
        start_pos (tuple): Starting position (x, y)
        goal_pos (tuple): Goal position (x, y)
        obstacles (set): Set of obstacle positions (x, y)
    
    Returns:
        bool: True if solvable, False otherwise
    """
    path_length, _ = a_star_pathfinding(grid_size_x, grid_size_y, start_pos, goal_pos, obstacles)
    return path_length is not None


def generate_solvable_maze(grid_size_x, grid_size_y, start_pos, goal_pos, 
                          obstacle_density=0.2, max_attempts=1000):
    """
    Generate a solvable maze with obstacles.
    
    Args:
        grid_size_x (int): Grid width
        grid_size_y (int): Grid height
        start_pos (tuple): Starting position (x, y)
        goal_pos (tuple): Goal position (x, y)
        obstacle_density (float): Fraction of cells to fill with obstacles (0.0 to 1.0)
        max_attempts (int): Maximum number of attempts to generate solvable maze
    
    Returns:
        set: Set of obstacle positions (x, y)
    """
    total_cells = grid_size_x * grid_size_y
    num_obstacles = int(total_cells * obstacle_density)
    
    # Ensure we don't place obstacles on start or goal
    forbidden_positions = {start_pos, goal_pos}
    
    for attempt in range(max_attempts):
        # Generate random obstacles
        all_positions = [(x, y) for x in range(grid_size_x) for y in range(grid_size_y)]
        available_positions = [pos for pos in all_positions if pos not in forbidden_positions]
        
        if len(available_positions) < num_obstacles:
            num_obstacles = len(available_positions)
        
        obstacles = set(random.sample(available_positions, num_obstacles))
        
        # Check if solvable
        if is_solvable(grid_size_x, grid_size_y, start_pos, goal_pos, obstacles):
            return obstacles
    
    # If we couldn't generate a solvable maze, return empty obstacles
    print(f"Warning: Could not generate solvable maze after {max_attempts} attempts. Using no obstacles.")
    return set()
