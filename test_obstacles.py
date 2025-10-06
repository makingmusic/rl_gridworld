#!/usr/bin/env python3
"""
Test script to demonstrate obstacle functionality in the gridworld environment.
"""

from gridworld2d import GridWorld2D, generate_solvable_maze, a_star_pathfinding
import numpy as np

def test_obstacle_functionality():
    """Test the obstacle functionality with a simple example."""
    
    print("=== Testing Obstacle Functionality ===\n")
    
    # Configuration
    grid_size_x = 8
    grid_size_y = 6
    start_pos = (0, 0)
    goal_pos = (grid_size_x - 1, grid_size_y - 1)
    obstacle_density = 0.2
    
    print(f"Grid size: {grid_size_x}x{grid_size_y}")
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    print(f"Obstacle density: {obstacle_density*100:.1f}%\n")
    
    # Generate solvable maze
    print("Generating solvable maze...")
    obstacles = generate_solvable_maze(
        grid_size_x, grid_size_y, start_pos, goal_pos, 
        obstacle_density=obstacle_density
    )
    print(f"Generated {len(obstacles)} obstacles: {sorted(obstacles)}\n")
    
    # Compute optimal path
    print("Computing optimal path using A*...")
    optimal_length, optimal_path = a_star_pathfinding(
        grid_size_x, grid_size_y, start_pos, goal_pos, obstacles
    )
    print(f"Optimal path length: {optimal_length} steps")
    print(f"Optimal path: {optimal_path}\n")
    
    # Create environment
    print("Creating environment...")
    env = GridWorld2D(
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        start_pos=start_pos,
        end_pos=goal_pos,
        obstacles=obstacles
    )
    
    # Test rendering
    print("Environment visualization:")
    print(env.render())
    print()
    
    # Test movement and collision detection
    print("Testing movement and collision detection...")
    env.reset()
    print(f"Starting position: {env.get_state()}")
    
    # Try to move into an obstacle
    test_actions = ["right", "up", "right", "up", "left", "down"]
    for action in test_actions:
        old_pos = env.get_state()
        next_state, reward, done = env.step(action)
        print(f"Action: {action:5} | {old_pos} -> {next_state} | Reward: {reward:6.3f} | Done: {done}")
        if done:
            break
    
    print(f"\nFinal position: {env.get_state()}")
    print(f"Reached goal: {env.is_terminal()}")
    
    # Test wall hit detection
    print("\nTesting wall hit detection...")
    env.reset()
    wall_hits = 0
    for _ in range(10):
        old_pos = env.get_state()
        next_state, reward, done = env.step("right")  # Try to go right repeatedly
        if next_state == old_pos and reward < 0:
            wall_hits += 1
        print(f"Position: {next_state} | Reward: {reward:6.3f} | Wall hits: {wall_hits}")
        if done:
            break
    
    print(f"\nTotal wall hits: {wall_hits}")
    print("\n=== Test completed successfully! ===")

if __name__ == "__main__":
    test_obstacle_functionality()
