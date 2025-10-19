from q_agent_nn import DQNAgent
from gridworld2d import GridWorld2D, generate_solvable_maze, a_star_pathfinding
import time
import plots
import matplotlib.pyplot as plt
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.console import Group, Console
from rich.panel import Panel
from rich.text import Text
import logWandB
import torch
import numpy as np

# Device selection logic for Apple Silicon (MPS), CUDA, or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# wandb parameters
USE_WANDB = False  # Set to False to disable wandb logging
N_IMAGE_EPISODES = 10  # Number of intermediate episodes to log with image

# Plot/display toggle
SHOW_PLOTS = False  # Set to False to disable matplotlib and grid/path visualizations

# Display/refresh controls
# Reduce terminal UI refresh frequency and NN forward-pass cadence for grid/path tables
LIVE_REFRESH_PER_SECOND = 1  # e.g., 1-5; lower = fewer UI redraws
DISPLAY_STEP_INTERVAL = 10000  # steps between NN-driven display updates; set 0/None to disable per-step updates
LEARNING_ACHIEVED_THRESHOLD = (
    5  # number of consecutive optimal episodes to confirm learning
)

# Neural Network specific parameters
learning_rate = 0.001  # learning rate for neural network (typically lower than tabular)
# buffer_size will be computed based on grid size - see compute_adaptive_buffer_size()
batch_size = 64  # batch size for neural network training
target_update_freq = 100  # frequency to update target network

# Q-learning parameters
discount_factor = 0.99  # discount factor for future rewards
epsilon = 1.0  # initial exploration rate
epsilon_decay = 0.999  # decay rate for exploration (slower decay for NN)
epsilon_min = 0.01  # minimum exploration rate
exploration_strategy = "epsilon_greedy"

# Grid Configuration Variables
num_episodes = 10000  # number of training episodes (more episodes for NN)
grid_size_x = 4  # width of the 2D grid
grid_size_y = 4  # height of the 2D grid
start_pos = (0, 0)  # starting position at bottom left
goal_pos = (grid_size_x - 1, grid_size_y - 1)  # goal position at top right

# Obstacle Configuration
USE_OBSTACLES = True  # Set to True to enable obstacles
OBSTACLE_DENSITY = 0.35  # Fraction of cells to fill with obstacles (0.0 to 1.0)


def compute_optimal_nn_size(grid_x, grid_y, min_hidden=128, max_hidden=2048):
    """
    Compute optimal neural network hidden layer size based on grid dimensions.
    Algorithm:
    1. Calculate total state space (grid_x * grid_y)
    2. Use adaptive parameters per state based on grid size:
       - Small grids (≤100 states): 10-15 params/state
       - Medium grids (100-2500 states): 15-20 params/state
       - Large grids (≥2500 states): 20-25 params/state
    3. Scale hidden size to achieve this ratio
    4. Apply reasonable bounds and round to power of 2

    Args:
        grid_x (int): Grid width
        grid_y (int): Grid height
        min_hidden (int): Minimum hidden layer size
        max_hidden (int): Maximum hidden layer size

    Returns:
        int: Optimal hidden layer size
    """
    total_states = grid_x * grid_y

    # Adaptive parameters per state based on grid size
    if total_states <= 100:  # Small grids (10x10)
        target_params_per_state = 12
    elif total_states <= 2500:  # Medium grids (up to 50x50)
        target_params_per_state = 18
    else:  # Large grids (100x100+)
        target_params_per_state = 25

    # For a 3-layer network: 2*h + h*h + h*h + h*4 ≈ 2*h^2 parameters
    # Solving: 2*h^2 ≈ target_params_per_state * total_states
    # h ≈ sqrt(target_params_per_state * total_states / 2)
    optimal_hidden = int(np.sqrt(target_params_per_state * total_states / 2))

    # Apply bounds (increased max for large grids)
    optimal_hidden = max(min_hidden, min(optimal_hidden, max_hidden))

    # Round to nearest power of 2 for computational efficiency
    optimal_hidden = 2 ** int(np.log2(optimal_hidden) + 0.5)

    return optimal_hidden


def compute_adaptive_buffer_size(grid_x, grid_y, base_size=10000):
    """
    Compute optimal replay buffer size based on grid dimensions.
    Larger grids need more diverse experiences for good learning.

    Strategy:
    - For small grids (10x10 = 100 states): use base_size
    - For medium grids (50x50 = 2500 states): use 2x base_size
    - For large grids (100x100 = 10000 states): use 4x base_size
    - Cap at reasonable maximum to avoid memory issues
    """
    total_states = grid_x * grid_y

    # Scale buffer size with grid area using square root scaling
    # This provides good coverage without excessive memory usage
    scale_factor = (total_states / 100) ** 0.5  # 100 is reference for 10x10 grid
    adaptive_size = base_size * scale_factor

    # Apply reasonable bounds: minimum 5k, maximum 200k
    adaptive_size = max(5000, min(adaptive_size, 200000))

    return int(adaptive_size)


def compute_adaptive_batch_size(grid_x, grid_y, base_size=64):
    """
    Compute optimal batch size based on grid dimensions.
    Larger grids can benefit from larger batches for more stable learning.
    """
    total_states = grid_x * grid_y
    # Scale batch size with grid area, but keep it reasonable
    adaptive_size = min(base_size * (total_states / 2500) ** 0.25, 256)
    return int(adaptive_size)


# Compute adaptive neural network parameters based on grid size
hidden_size = compute_optimal_nn_size(grid_size_x, grid_size_y)
buffer_size = compute_adaptive_buffer_size(grid_size_x, grid_size_y)
batch_size = compute_adaptive_batch_size(grid_size_x, grid_size_y)

# Display computed parameters
print(f"Grid size: {grid_size_x}x{grid_size_y} ({grid_size_x * grid_size_y} states)")
print(f"Computed NN hidden size: {hidden_size}")
print(f"Computed buffer size: {buffer_size}")
print(f"Computed batch size: {batch_size}")
print(f"Estimated NN parameters: ~{2 * hidden_size**2 + hidden_size * 4}")
print(
    f"Parameters per state: ~{(2 * hidden_size**2 + hidden_size * 4) / (grid_size_x * grid_size_y):.1f}"
)


def compute_max_steps(grid_x, grid_y, epsilon, recent_steps=None):
    short_cut = 50 * (grid_x + grid_y)
    D = (grid_x - 1) + (grid_y - 1)
    area = grid_x * grid_y
    c_area = 2.0
    max_limit = 30 * (grid_x + grid_y)

    # Adaptive branch when recent history is available
    if recent_steps is not None and len(recent_steps) > 0:
        # Use a robust statistic (75th percentile) to avoid outliers growing the cap
        try:
            p75 = float(np.percentile(recent_steps, 75))
        except Exception:
            p75 = sum(recent_steps) / len(recent_steps)
        cap = max(2 * D, 1.2 * p75)
        cap = min(cap, c_area * area)
        cap = min(cap, max_limit)
        return int(short_cut)
        # return int(cap)

    # Epsilon-aware default
    k0, k_eps = 2.0, 4.0
    cap = D * (k0 + k_eps * epsilon)

    # Clamp within [2D, c_area * area]
    cap = min(max(cap, 2 * D), c_area * area)
    cap = min(cap, max_limit)
    return int(cap)


max_steps_per_episode = compute_max_steps(grid_size_x, grid_size_y, epsilon)

# display parameters
sleep_time = 0  # time to sleep between episodes

print("Initializing Deep Q-Network Agent...")
print(f"Using device: {device}")

# Generate obstacles if enabled
obstacles = set()
optimal_path_length = None
if USE_OBSTACLES:
    print(
        f"Generating solvable maze with {OBSTACLE_DENSITY * 100:.1f}% obstacle density..."
    )
    obstacles = generate_solvable_maze(
        grid_size_x, grid_size_y, start_pos, goal_pos, obstacle_density=OBSTACLE_DENSITY
    )
    print(f"Generated {len(obstacles)} obstacles")

    # Compute optimal path length using A*
    optimal_path_length, optimal_path = a_star_pathfinding(
        grid_size_x, grid_size_y, start_pos, goal_pos, obstacles
    )
    print(f"Optimal path length: {optimal_path_length} steps")
else:
    # No obstacles - optimal path is Manhattan distance
    optimal_path_length = (grid_size_x - 1) + (grid_size_y - 1)
    print(f"No obstacles - optimal path length: {optimal_path_length} steps")

# Initialize environment and agent
env = GridWorld2D(
    grid_size_x=grid_size_x,
    grid_size_y=grid_size_y,
    start_pos=start_pos,
    end_pos=goal_pos,
    max_steps_per_episode=max_steps_per_episode,
    obstacles=obstacles,
)

agent = DQNAgent(
    actions=["up", "down", "left", "right"],
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_min=epsilon_min,
    exploration_strategy=exploration_strategy,
    grid_size_x=grid_size_x,
    grid_size_y=grid_size_y,
    buffer_size=buffer_size,
    batch_size=batch_size,
    target_update_freq=target_update_freq,
    hidden_size=hidden_size,
    device=device,  # Pass device to agent
)

# Initialize matplotlib figures and axes for all visualizations
if SHOW_PLOTS:
    qtable_fig, qtable_ax = plt.subplots(figsize=(grid_size_x, grid_size_y))
    steps_fig, steps_ax = plt.subplots(figsize=(8, 6))
    epsilon_fig, epsilon_ax = plt.subplots(figsize=(8, 6))

# Initialize lists to store episode and step data
episode_data = []
step_data = []
epsilon_data = []
last_ten_steps = ["0"] * 10  # Store steps (with markers) from last 10 episodes
last_ten_steps_numeric = [0] * 10  # Numeric steps history for adaptive cap
last_ten_rewards = ["0.00"] * 10  # Store rewards from last 10 episodes
last_ten_rewards_numeric = [0.0] * 10  # Numeric rewards history

# Initialize tracking variables for obstacle metrics
success_count = 0  # Number of successful episodes
wall_hit_count = 0  # Number of wall hits
total_wall_hits = 0  # Total wall hits across all episodes
total_returns = 0.0  # Total returns across all episodes
suboptimality_ratios = []  # List of suboptimality ratios per episode

# Learning detection variables
consecutive_optimal_episodes = 0  # Track consecutive episodes with optimal path
learning_achieved_episode = None  # Episode when learning was first achieved
learning_detection_threshold = LEARNING_ACHIEVED_THRESHOLD

# Display control for large grids
SHOW_GRID_DISPLAYS = (
    grid_size_x * grid_size_y
) <= 1000  # Hide grid displays for large grids

# Initialize the progress bars (always shown)
progress = Progress(
    SpinnerColumn(),
    TextColumn("Training DQN progress:"),
    BarColumn(complete_style="blue", finished_style="green"),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TaskProgressColumn(),
    transient=True,
)
task = progress.add_task("Training Deep Q-Network agent", total=num_episodes)

# Add position progress bar
posProgressBar = Progress(
    SpinnerColumn(),
    TextColumn("Current position:"),
    BarColumn(complete_style="yellow", finished_style="green"),
    MofNCompleteColumn(),
    transient=True,
)
posTask = posProgressBar.add_task(
    "Position tracking", total=grid_size_x * grid_size_y - 1
)

# Add steps progress bar
stepsProgressBar = Progress(
    SpinnerColumn(),
    TextColumn("Steps in last 10 episodes ({task.fields[max_cap]}):"),
    TextColumn("[bold blue]{task.description}"),
    TextColumn(" | Current:"),
    TextColumn("[bold green]{task.fields[current_steps]}"),
    transient=True,
)
stepsTask = stepsProgressBar.add_task(
    "Steps tracking", total=0, current_steps=0, max_cap=env.max_steps_per_episode
)

# Add rewards progress bar
rewardsProgressBar = Progress(
    SpinnerColumn(),
    TextColumn("Rewards in last 10 episodes:"),
    TextColumn("[bold blue]{task.description}"),
    TextColumn(" | Current:"),
    TextColumn("[bold green]{task.fields[current_reward]}"),
    transient=True,
)
rewardsTask = rewardsProgressBar.add_task(
    "Rewards tracking", total=0, current_reward=0.0
)

# Initialize grid/path rich tables (only for small grids)
grid_display = None
path_display = None
if SHOW_GRID_DISPLAYS:
    grid_display = plots.create_grid_display(
        grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable(), obstacles
    )
    _saved_eps = agent.epsilon
    agent.epsilon = 0.0
    path_display = plots.display_actual_path(
        grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable(), obstacles
    )
    agent.epsilon = _saved_eps

# Initialize NN training notification state
nn_training_note = Text("", style="bold magenta")

# Initialize learning progress notification
learning_progress_note = Text("", style="bold green")

# Initialize reward display notification
reward_display_note = Text("", style="bold cyan")

# Build initial display group including tables
display_components = [
    progress,
    posProgressBar,
    stepsProgressBar,
    rewardsProgressBar,
    Panel(nn_training_note, title="NN Training", border_style="magenta"),
    Panel(learning_progress_note, title="Learning Progress", border_style="green"),
    Panel(reward_display_note, title="Episode Reward", border_style="cyan"),
]

# Add grid displays only for small grids
if SHOW_GRID_DISPLAYS:
    grid_table = plots.grid_to_table(grid_display)
    display_components.extend(
        [
            Panel(grid_table, title="Grid (DQN)"),
            Panel(path_display, title="Current Best Path"),
        ]
    )

display_group = Group(*display_components)

# Create base display group (textual components always shown)
# display_group = Group(
#     progress,
#     posProgressBar,
#     stepsProgressBar,
#     Panel(nn_training_note, title="NN Training", border_style="magenta"),
# )

# Initialize wandb
if USE_WANDB:
    wandbconfig = {
        "agent_type": "DQN",
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min,
        "num_episodes": num_episodes,
        "grid_size_x": grid_size_x,
        "grid_size_y": grid_size_y,
        "start_pos": start_pos,
        "goal_pos": goal_pos,
        "exploration_strategy": exploration_strategy,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "target_update_freq": target_update_freq,
        "hidden_size": hidden_size,
        "device": str(agent.device),
        # Additional computed parameters for analysis
        "total_states": grid_size_x * grid_size_y,
        "estimated_nn_parameters": 2 * hidden_size**2 + hidden_size * 4,
        "parameters_per_state": (2 * hidden_size**2 + hidden_size * 4)
        / (grid_size_x * grid_size_y),
        "adaptive_sizing": True,
        # Obstacle parameters
        "use_obstacles": USE_OBSTACLES,
        "obstacle_density": OBSTACLE_DENSITY,
        "num_obstacles": len(obstacles),
        "optimal_path_length": optimal_path_length,
    }
    logWandB.initWandB("rl-gridworld-dqn", config=wandbconfig)

with Live(
    display_group, refresh_per_second=LIVE_REFRESH_PER_SECOND
) as live:  # Always keep textual UI active
    # Training loop
    start_time = time.time()
    for episode in range(num_episodes):
        # Recompute adaptive max steps using recent numeric history and current epsilon
        # Make the cap non-increasing to avoid positive feedback loops
        try:
            adaptive_cap = compute_max_steps(
                grid_size_x,
                grid_size_y,
                agent.epsilon,
                recent_steps=last_ten_steps_numeric,
            )
        except Exception:
            adaptive_cap = compute_max_steps(grid_size_x, grid_size_y, agent.epsilon)

        if episode == 0 or getattr(env, "max_steps_per_episode", None) is None:
            env.max_steps_per_episode = adaptive_cap
        else:
            env.max_steps_per_episode = min(env.max_steps_per_episode, adaptive_cap)

        state = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        episode_start_time = time.time()  # Track episode start time
        episode_wall_hits = 0  # Track wall hits in this episode

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # Track wall hits (when agent stays in same position and gets negative reward)
            if next_state == state and reward < 0:
                episode_wall_hits += 1
                total_wall_hits += 1

            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
            episode_reward += reward

            # If a NN training just occurred, update the main display note
            if getattr(agent, "just_trained", False):
                nn_training_note = Text(
                    f"NN training triggered (total: {agent.training_runs})",
                    style="bold magenta",
                )
                # reset the flag so message is only refreshed on new trainings
                agent.just_trained = False

            # Update position progress bar
            current_pos = env.get_state()
            pos_value = current_pos[0] * grid_size_y + current_pos[1]
            posProgressBar.update(posTask, completed=pos_value)

            # Update steps progress bar
            steps_str = " | ".join(last_ten_steps)
            stepsProgressBar.update(
                stepsTask,
                description=steps_str,
                current_steps=step_count,
                max_cap=env.max_steps_per_episode,
            )

            # Update rewards progress bar
            rewards_str = " | ".join(last_ten_rewards)
            rewardsProgressBar.update(
                rewardsTask,
                description=rewards_str,
                current_reward=episode_reward,
            )

            # Update grid/path rich tables periodically (not every step for performance)
            should_update_display = (
                bool(DISPLAY_STEP_INTERVAL)
                and DISPLAY_STEP_INTERVAL > 0
                and (step_count % DISPLAY_STEP_INTERVAL == 0)
            ) or done

            if should_update_display:
                # Cache Q-table once per update to avoid duplicate forward passes
                qtable_cached = agent.getQTable()

                display_components = [
                    progress,
                    posProgressBar,
                    stepsProgressBar,
                    rewardsProgressBar,
                    Panel(
                        nn_training_note, title="NN Training", border_style="magenta"
                    ),
                    Panel(
                        learning_progress_note,
                        title="Learning Progress",
                        border_style="green",
                    ),
                    Panel(
                        reward_display_note,
                        title="Episode Reward",
                        border_style="cyan",
                    ),
                ]

                # Add grid displays only for small grids
                if SHOW_GRID_DISPLAYS and grid_display is not None:
                    grid_display = plots.update_grid_display(
                        grid_display, qtable_cached, start_pos, goal_pos, obstacles
                    )
                    grid_table = plots.grid_to_table(grid_display)
                    display_components.extend(
                        [
                            Panel(grid_table, title="Grid (DQN)"),
                            # Recompute Current Best Path greedily (epsilon=0)
                            Panel(
                                plots.display_actual_path(
                                    grid_size_x,
                                    grid_size_y,
                                    start_pos,
                                    goal_pos,
                                    qtable_cached,
                                    obstacles,
                                ),
                                title="Current Best Path",
                            ),
                        ]
                    )

                display_group = Group(*display_components)
                live.update(display_group)

        # Store episode and step data
        episode_data.append(episode)
        step_data.append(step_count)
        epsilon_data.append(agent.epsilon)

        # Track success and metrics
        episode_success = done and env.is_terminal()
        if episode_success:
            success_count += 1

        total_returns += episode_reward

        # Calculate suboptimality ratio
        if episode_success and optimal_path_length is not None:
            suboptimality_ratio = step_count / optimal_path_length
            suboptimality_ratios.append(suboptimality_ratio)
        else:
            suboptimality_ratios.append(None)

        # Learning detection: Check if agent achieved optimal path
        if not episode_success:  # Episode didn't complete successfully
            consecutive_optimal_episodes = 0  # Reset counter if episode didn't complete
        else:
            # Check if the path taken was optimal
            best_path_length = plots.get_best_path_length(
                grid_size_x,
                grid_size_y,
                start_pos,
                goal_pos,
                agent.getQTable(),
                obstacles,
            )

            if best_path_length is not None and best_path_length <= optimal_path_length:
                consecutive_optimal_episodes += 1
                if learning_achieved_episode is None:
                    learning_achieved_episode = episode
            else:
                consecutive_optimal_episodes = 0  # Reset counter if path wasn't optimal

        # Update learning progress display
        if learning_achieved_episode is not None:
            learning_progress_note = Text(
                f"First learning achieved at episode {learning_achieved_episode + 1} | "
                f"Consecutive optimal: {consecutive_optimal_episodes}/{learning_detection_threshold}",
                style="bold green",
            )
        else:
            learning_progress_note = Text(
                f"Consecutive optimal episodes: {consecutive_optimal_episodes}/{learning_detection_threshold} | "
                f"Target: {optimal_path_length} steps",
                style="bold yellow",
            )

        # Update reward display
        reward_display_note = Text(
            f"Episode {episode + 1} Reward: {episode_reward:.2f} | "
            f"Total Steps: {step_count} | "
            f"Success: {'Yes' if episode_success else 'No'}",
            style="bold cyan",
        )

        # Early stopping: Check if learning has been achieved
        if consecutive_optimal_episodes >= learning_detection_threshold:
            print("\n🎉 LEARNING ACHIEVED! 🎉")
            print("Neural network successfully learned the optimal path!")
            print(
                f"First Learning achieved at episode: {learning_achieved_episode + 1}"
            )
            print(f"Consecutive optimal episodes: {consecutive_optimal_episodes}")
            print(f"Optimal path length: {optimal_path_length} steps")
            print(f"Training stopped early after {episode + 1} episodes")
            break

        # Log metrics to wandb
        if USE_WANDB and plots.shouldThisEpisodeBeLogged(
            episode, num_episodes, N_IMAGE_EPISODES
        ):
            episode_duration = time.time() - episode_start_time
            best_path_length = plots.get_best_path_length(
                grid_size_x,
                grid_size_y,
                start_pos,
                goal_pos,
                agent.getQTable(),
                obstacles,
            )
            if SHOW_PLOTS:
                q_table_img, qtable_fig, qtable_ax = plots.saveQTableAsImage(
                    agent.getQTableAsPolicyArrows(),
                    filename=None,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    fig=qtable_fig,
                    ax=qtable_ax,
                )

            # Calculate metrics
            success_rate = success_count / (episode + 1)
            mean_steps = np.mean(step_data) if step_data else 0
            wall_hit_rate = total_wall_hits / (episode + 1) if episode > 0 else 0
            mean_return = total_returns / (episode + 1) if episode > 0 else 0
            current_suboptimality = (
                suboptimality_ratios[-1]
                if suboptimality_ratios and suboptimality_ratios[-1] is not None
                else None
            )

            wandbconfig = {
                "episode": episode,
                "steps": step_count,
                "epsilon": agent.epsilon,
                "reward": episode_reward,
                "episode_duration": episode_duration,
                "best_path_length": best_path_length,
                "replay_buffer_size": len(agent.memory),
                "training_step": agent.training_step,
                "success_rate": success_rate,
                "mean_steps": mean_steps,
                "wall_hit_rate": wall_hit_rate,
                "mean_return": mean_return,
                "suboptimality_ratio": current_suboptimality,
                "episode_wall_hits": episode_wall_hits,
                "optimal_path_length": optimal_path_length,
            }
            logWandB.logEpisodeWithImageControl(
                wandbconfig,
                step=episode,
                episode=episode,
                total_episodes=num_episodes,
                N=N_IMAGE_EPISODES,
            )
            if SHOW_PLOTS:
                # Update plots at the end of each episode
                steps_fig, steps_ax = plots.plotStepsPerEpisode(
                    plt, episode_data, step_data, fig=steps_fig, ax=steps_ax
                )
                epsilon_fig, epsilon_ax = plots.plotEpsilonDecayPerEpisode(
                    plt, episode_data, epsilon_data, fig=epsilon_fig, ax=epsilon_ax
                )

        # Update last ten steps with (M) marker if episode hit max steps
        timed_out = (step_count >= env.max_steps_per_episode) and (
            not env.is_terminal()
        )
        # Update numeric and display histories
        last_ten_steps_numeric.pop(0)
        last_ten_steps_numeric.append(step_count)
        last_ten_steps.pop(0)
        last_ten_steps.append(f"{step_count}{'(M)' if timed_out else ''}")

        # Update reward histories
        last_ten_rewards_numeric.pop(0)
        last_ten_rewards_numeric.append(episode_reward)
        last_ten_rewards.pop(0)
        last_ten_rewards.append(f"{episode_reward:.2f}")

        # Get the Q-table for this episode (approximate from neural network)
        qtable = agent.getQTable()
        #        episode_qtable = {}
        #        for state, actions in qtable.items():
        #            episode_qtable[state] = {"up": actions["up"], "down": actions["down"], "left": actions["left"], "right": actions["right"],}

        # Decay exploration rate at end of episode
        agent.decay_epsilon()

        # Update Displays
        progress.update(
            task,
            description=f"Episode {episode + 1} of {num_episodes} (DQN)",
            advance=1,
        )
        live.update(display_group)
        time.sleep(sleep_time)

        # Periodic refresh of path and grid tables (less frequent for performance)
        if episode % 50 == 0 or episode == num_episodes - 1:
            display_components = [
                progress,
                posProgressBar,
                stepsProgressBar,
                rewardsProgressBar,
                Panel(nn_training_note, title="NN Training", border_style="magenta"),
                Panel(
                    learning_progress_note,
                    title="Learning Progress",
                    border_style="green",
                ),
                Panel(
                    reward_display_note,
                    title="Episode Reward",
                    border_style="cyan",
                ),
            ]

            # Add grid displays only for small grids
            if SHOW_GRID_DISPLAYS and grid_display is not None:
                _saved_eps = agent.epsilon
                agent.epsilon = 0.0
                path_display = plots.display_actual_path(
                    grid_size_x,
                    grid_size_y,
                    start_pos,
                    goal_pos,
                    agent.getQTable(),
                    obstacles,
                )
                agent.epsilon = _saved_eps
                grid_display = plots.update_grid_display(
                    grid_display, agent.getQTable(), start_pos, goal_pos, obstacles
                )
                grid_table = plots.grid_to_table(grid_display)
                display_components.extend(
                    [
                        Panel(grid_table, title="Grid (DQN)"),
                        Panel(path_display, title="Current Best Path"),
                    ]
                )

            display_group = Group(*display_components)
            live.update(display_group)

    end_time = time.time()

# Final display update to show completion
progress.update(task, description="DQN Training completed", completed=num_episodes)

display_components = [
    progress,
    posProgressBar,
    stepsProgressBar,
    rewardsProgressBar,
    Panel(nn_training_note, title="NN Training", border_style="magenta"),
    Panel(learning_progress_note, title="Learning Progress", border_style="green"),
    Panel(reward_display_note, title="Episode Reward", border_style="cyan"),
]

# Add grid displays only for small grids
if SHOW_GRID_DISPLAYS and grid_display is not None:
    grid_display = plots.update_grid_display(
        grid_display, agent.getQTable(), start_pos, goal_pos, obstacles
    )
    grid_table = plots.grid_to_table(grid_display)
    display_components.extend(
        [
            Panel(grid_table, title="Grid (DQN)"),
            Panel(path_display, title="Current Best Path"),
        ]
    )

display_group = Group(*display_components)
live.update(display_group)

print(f"DQN Training completed in {end_time - start_time:.2f} seconds")


def evaluate_agent_greedy(agent, env, grid_size_x, grid_size_y, num_episodes=5):
    """
    Evaluate the agent with epsilon=0 (greedy policy) and no learning.
    Returns a list of steps taken per episode.
    """
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    eval_steps = []
    max_steps = grid_size_x * grid_size_y * 4  # Safety cap to avoid infinite loops

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps_taken = 0
        while not done and steps_taken < max_steps:
            action = agent.choose_action(state)
            next_state, _, done = env.step(action)
            state = next_state
            steps_taken += 1
        eval_steps.append(steps_taken)

    agent.epsilon = saved_epsilon
    print(
        f"Evaluation with epsilon=0 → steps per episode: {eval_steps} | avg: {np.mean(eval_steps):.2f}"
    )
    return eval_steps


# Run evaluation episodes with epsilon = 0 (greedy) and no learning
eval_steps = evaluate_agent_greedy(agent, env, grid_size_x, grid_size_y, num_episodes=5)

# Display the actual path taken by the model (always show text-based table)
# Final path display under greedy policy
_saved_eps = agent.epsilon
agent.epsilon = 0.0
path_table = plots.display_actual_path(
    grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable(), obstacles
)
agent.epsilon = _saved_eps

if SHOW_PLOTS and USE_WANDB:
    q_table_policyarrows = agent.getQTableAsPolicyArrows()
    q_table_img, qtable_fig, qtable_ax = plots.saveQTableAsImage(
        q_table_policyarrows,
        "q_heatmap_dqn.png",
        start_pos,
        goal_pos,
        fig=qtable_fig,
        ax=qtable_ax,
    )
    wandbconfig = {"q_table_heatmap_img": logWandB.wandb.Image(q_table_img)}
    logWandB.logEpisode(wandbconfig, step=num_episodes - 1)

if SHOW_PLOTS:
    # Clean up all matplotlib figures
    plt.close(qtable_fig)
    plt.close(steps_fig)
    plt.close(epsilon_fig)

console = Console()
console.print("\nActual path taken by the DQN model:")
console.print(Panel(path_table, title="Path"))

if USE_WANDB:
    logWandB.closeWandB()

if SHOW_PLOTS:
    # Plot final visualizations
    steps_fig, steps_ax = plots.plotStepsPerEpisode(
        plt, episode_data, step_data, fig=steps_fig, ax=steps_ax
    )
    epsilon_fig, epsilon_ax = plots.plotEpsilonDecayPerEpisode(
        plt, episode_data, epsilon_data, fig=epsilon_fig, ax=epsilon_ax
    )

# Print final Q-table approximation for debugging
# print("\nFinal Q-Table approximation from DQN:")
# agent.print_q_table()

print("\nFinal training statistics:")
print(f"Total episodes: {len(episode_data)}")
print(f"Final epsilon: {agent.epsilon:.4f}")
print(f"Final replay buffer size: {len(agent.memory)}")
print(f"Total training steps: {agent.training_step}")
print(f"Average steps per episode (last 100): {np.mean(step_data[-100:]):.2f}")

# Obstacle-specific metrics
if USE_OBSTACLES:
    print(f"\nObstacle Navigation Metrics:")
    print(f"Success rate: {success_count / len(episode_data) * 100:.1f}%")
    print(f"Mean steps per episode: {np.mean(step_data):.2f}")
    print(f"Wall hit rate: {total_wall_hits / len(episode_data):.2f} hits per episode")
    print(f"Mean return per episode: {total_returns / len(episode_data):.2f}")
    print(f"Optimal path length: {optimal_path_length} steps")

    # Calculate mean suboptimality ratio for successful episodes
    successful_suboptimality = [
        ratio for ratio in suboptimality_ratios if ratio is not None
    ]
    if successful_suboptimality:
        mean_suboptimality = np.mean(successful_suboptimality)
        print(
            f"Mean suboptimality ratio: {mean_suboptimality:.2f} (agent_steps / optimal_steps)"
        )
        print(f"Best suboptimality ratio: {min(successful_suboptimality):.2f}")
    else:
        print("No successful episodes to calculate suboptimality ratio")

# Learning achievement reporting
if learning_achieved_episode is not None:
    print("\n🎯 LEARNING ACHIEVEMENT SUMMARY:")
    print("✅ Neural network successfully learned the optimal path!")
    print(f"📊 Learning achieved at episode: {learning_achieved_episode + 1}")
    print(f"📈 Consecutive optimal episodes achieved: {consecutive_optimal_episodes}")
    print(f"🎯 Optimal path length: {optimal_path_length} steps")
    print(
        f"⏱️  Training completed in {len(episode_data)} episodes (out of {num_episodes} planned)"
    )
else:
    print("\n❌ LEARNING NOT ACHIEVED:")
    print(
        f"Neural network did not achieve optimal path in {len(episode_data)} episodes"
    )
    print(f"Optimal path length target: {optimal_path_length} steps")
    print(f"Final consecutive optimal episodes: {consecutive_optimal_episodes}")

# Plot steps per episode : Uncomment this section to see the steps per episode
# plots.plotStepsPerEpisode(plt, episode_data, step_data)
# plt.tight_layout()
# plt.show()
