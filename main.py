from q_agent2D import QLearningAgent2D
from gridworld2d import GridWorld2D
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
import logWandB
import pandas as pd
import numpy as np

# wandb parameters
USE_WANDB = False  # Set to False to disable wandb logging
N_IMAGE_EPISODES = 10  # Number of intermediate episodes to log with image

# Plot/display toggle
SHOW_PLOTS = False  # Set to False to disable matplotlib and grid/path visualizations

# Early stopping parameters
LEARNING_ACHIEVED_THRESHOLD = (
    5  # number of consecutive optimal episodes to confirm learning
)


# default values for learning parameters
learning_rate = 0.1  # learning rate for Q-value updates
discount_factor = 0.99  # discount factor for future rewards
epsilon = 1.0  # initial exploration rate
epsilon_decay = 0.95  # decay rate for exploration
epsilon_min = 0.01  # minimum exploration rate
# Epsilon-greedy parameters
exploration_strategy = "epsilon_greedy"
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
# Grid Configuration Variables
num_episodes = 10000  # number of training episodes
grid_size_x = 20  # width of the 2D grid
grid_size_y = 20  # height of the 2D grid
start_pos = (0, 0)  # starting position at bottom left
goal_pos = (grid_size_x - 1, grid_size_y - 1)  # goal position at top right


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


# Calculate optimal path length (Manhattan distance for simple grid)
optimal_path_length = (grid_size_x - 1) + (grid_size_y - 1)
print(f"Optimal path length: {optimal_path_length} steps")

# Initialize environment and agent
env = GridWorld2D(
    grid_size_x=grid_size_x,
    grid_size_y=grid_size_y,
    start_pos=start_pos,
    end_pos=goal_pos,
    max_steps_per_episode=max_steps_per_episode,
)
agent = QLearningAgent2D(
    actions=["up", "down", "left", "right"],
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_min=epsilon_min,
    exploration_strategy=exploration_strategy,
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
qtable_data = []
last_ten_steps = ["0"] * 10  # Store steps (with markers) from last 10 episodes
last_ten_steps_numeric = [0] * 10  # Numeric steps history for adaptive cap

# Learning detection variables
consecutive_optimal_episodes = 0  # Track consecutive episodes with optimal path
learning_achieved_episode = None  # Episode when learning was first achieved
learning_detection_threshold = LEARNING_ACHIEVED_THRESHOLD

# Initialize the progress bars
progress = Progress(
    SpinnerColumn(),
    TextColumn("Training progress:"),
    BarColumn(complete_style="blue", finished_style="green"),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TaskProgressColumn(),
    transient=True,
)
task = progress.add_task("Training Q-learning agent", total=num_episodes)

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

# Initialize grid display
grid_display = plots.create_grid_display(
    grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
)
# Ensure greedy evaluation (epsilon=0) when showing Current Best Path
_saved_eps = agent.epsilon
agent.epsilon = 0.0
path_display = plots.display_actual_path(
    grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
)
agent.epsilon = _saved_eps

# Create display group with progress bars first
display_group = Group(progress, posProgressBar, stepsProgressBar)

# Initialize wandb
if USE_WANDB:
    wandbconfig = {
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
        "optimal_path_length": optimal_path_length,
        "learning_achieved_threshold": LEARNING_ACHIEVED_THRESHOLD,
    }
    logWandB.initWandB("rl-gridworld-qlearning", config=wandbconfig)

with Live(display_group, refresh_per_second=50) as live:
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

        qtable_data.append(episode)
        state = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        episode_start_time = time.time()  # Track episode start time

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
            episode_reward += reward

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

            # Update grid display and show it after progress bars
            grid_display = plots.update_grid_display(
                grid_display, agent.getQTable(), start_pos, goal_pos
            )
            grid_table = plots.grid_to_table(grid_display)

            # Use epsilon=0 for greedy evaluation of Current Best Path
            _saved_eps = agent.epsilon
            agent.epsilon = 0.0
            current_best_path = plots.display_actual_path(
                grid_size_x,
                grid_size_y,
                start_pos,
                goal_pos,
                agent.getQTable(),
            )
            agent.epsilon = _saved_eps
            
            display_group = Group(
                progress,
                posProgressBar,
                stepsProgressBar,
                Panel(grid_table, title="Grid"),
                Panel(current_best_path, title="Current Best Path"),
            )
            live.update(display_group)

        # Store episode and step data
        episode_data.append(episode)
        step_data.append(step_count)
        epsilon_data.append(agent.epsilon)
        
        # Learning detection: Check if agent achieved optimal path
        episode_success = done and env.is_terminal()
        if not episode_success:  # Episode didn't complete successfully
            consecutive_optimal_episodes = 0  # Reset counter if episode didn't complete
        else:
            # Check if the path taken was optimal
            # Use epsilon=0 for greedy evaluation
            _saved_eps = agent.epsilon
            agent.epsilon = 0.0
            best_path_length = plots.get_best_path_length(
                grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
            )
            agent.epsilon = _saved_eps

            if best_path_length is not None and best_path_length <= optimal_path_length:
                consecutive_optimal_episodes += 1
                if learning_achieved_episode is None:
                    learning_achieved_episode = episode
            else:
                consecutive_optimal_episodes = 0  # Reset counter if path wasn't optimal

        # Early stopping: Check if learning has been achieved
        if consecutive_optimal_episodes >= learning_detection_threshold:
            print("\n🎉 LEARNING ACHIEVED! 🎉")
            print("Q-learning agent successfully learned the optimal path!")
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
            # Use epsilon=0 for greedy evaluation
            _saved_eps = agent.epsilon
            agent.epsilon = 0.0
            best_path_length = plots.get_best_path_length(
                grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
            )
            agent.epsilon = _saved_eps
            if SHOW_PLOTS:
                q_table_img, qtable_fig, qtable_ax = plots.saveQTableAsImage(
                    agent.getQTableAsPolicyArrows(),
                    filename=None,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    fig=qtable_fig,
                    ax=qtable_ax,
                )
            q_table_df = pd.DataFrame.from_dict(
                {k: v for k, v in agent.getQTable().items()}, orient="index"
            )
            wandbconfig = {
                "episode": episode,
                "steps": step_count,
                "epsilon": agent.epsilon,
                "reward": episode_reward,
                "episode_duration": episode_duration,
                "best_path_length": best_path_length,
                "consecutive_optimal_episodes": consecutive_optimal_episodes,
                "learning_achieved": learning_achieved_episode is not None,
                "episode_success": episode_success,
            }
            logWandB.logEpisodeWithImageControl(
                wandbconfig,
                step=episode,
                episode=episode,
                total_episodes=num_episodes,
                N=N_IMAGE_EPISODES,
            )

            # Update plots at the end of each episode
            if SHOW_PLOTS:
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
        last_ten_steps.pop(0)  # Remove oldest step count
        last_ten_steps.append(
            f"{step_count}{'(M)' if timed_out else ''}"
        )  # Add current step count

        # Get the Q-table for this episode
        qtable = agent.getQTable()
        episode_qtable = {}
        for state, actions in qtable.items():
            episode_qtable[state] = {
                "up": actions["up"],
                "down": actions["down"],
                "left": actions["left"],
                "right": actions["right"],
            }
        qtable_data.append(episode_qtable)

        # Decay exploration rate at end of episode
        agent.decay_epsilon()

        # Update Displays
        progress.update(
            task, description=f"Episode {episode + 1} of {num_episodes}", advance=1
        )
        live.update(display_group)
        time.sleep(sleep_time)

        # Update path display at the end of each episode
        # Ensure greedy evaluation (epsilon=0) when updating Current Best Path
        _saved_eps = agent.epsilon
        agent.epsilon = 0.0
        path_display = plots.display_actual_path(
            grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
        )
        agent.epsilon = _saved_eps
        display_group = Group(
            progress,
            posProgressBar,
            stepsProgressBar,
            Panel(grid_table, title="Grid"),
            Panel(path_display, title="Current Best Path"),
        )
        live.update(display_group)

    end_time = time.time()

# Final display update to show completion
progress.update(task, description="Training completed", completed=num_episodes)
grid_display = plots.update_grid_display(
    grid_display, agent.getQTable(), start_pos, goal_pos
)
live.update(display_group)
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Learning achievement reporting
if learning_achieved_episode is not None:
    print("\n🎯 LEARNING ACHIEVEMENT SUMMARY:")
    print("✅ Q-learning agent successfully learned the optimal path!")
    print(f"📊 Learning achieved at episode: {learning_achieved_episode + 1}")
    print(f"📈 Consecutive optimal episodes achieved: {consecutive_optimal_episodes}")
    print(f"🎯 Optimal path length: {optimal_path_length} steps")
    print(
        f"⏱️  Training completed in {len(episode_data)} episodes (out of {num_episodes} planned)"
    )
else:
    print("\n❌ LEARNING NOT ACHIEVED:")
    print(
        f"Q-learning agent did not achieve optimal path in {len(episode_data)} episodes"
    )
    print(f"Optimal path length target: {optimal_path_length} steps")
    print(f"Final consecutive optimal episodes: {consecutive_optimal_episodes}")

# Display the actual path taken by the model
# Final path display under greedy policy
_saved_eps = agent.epsilon
agent.epsilon = 0.0
path_table = plots.display_actual_path(
    grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
)
agent.epsilon = _saved_eps

if USE_WANDB and SHOW_PLOTS:
    q_table_policyarrows = agent.getQTableAsPolicyArrows()
    q_table_img, qtable_fig, qtable_ax = plots.saveQTableAsImage(
        q_table_policyarrows,
        "q_heatmap.png",
        start_pos,
        goal_pos,
        fig=qtable_fig,
        ax=qtable_ax,
    )
    wandbconfig = {"q_table_heatmap_img": logWandB.wandb.Image(q_table_img)}
    logWandB.logEpisode(wandbconfig, step=num_episodes - 1)

# Clean up all matplotlib figures
if SHOW_PLOTS:
    plt.close(qtable_fig)
    plt.close(steps_fig)
    plt.close(epsilon_fig)

console = Console()
console.print("\nActual path taken by the model:")
console.print(Panel(path_table, title="Path"))

if USE_WANDB:
    logWandB.closeWandB()

# Plot final visualizations
if SHOW_PLOTS:
    steps_fig, steps_ax = plots.plotStepsPerEpisode(
        plt, episode_data, step_data, fig=steps_fig, ax=steps_ax
    )
    epsilon_fig, epsilon_ax = plots.plotEpsilonDecayPerEpisode(
        plt, episode_data, epsilon_data, fig=epsilon_fig, ax=epsilon_ax
    )

# Plot steps per episode : Uncomment this section to see the steps per episode
# plots.plotStepsPerEpisode(plt, episode_data, step_data)
# plt.tight_layout()
# plt.show()
