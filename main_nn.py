from q_agent_nn import DQNAgent
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

# Neural Network specific parameters
learning_rate = 0.001  # learning rate for neural network (typically lower than tabular)
buffer_size = 10000  # experience replay buffer size
batch_size = 64  # batch size for neural network training
target_update_freq = 100  # frequency to update target network
hidden_size = 128  # size of hidden layers

# Q-learning parameters
discount_factor = 0.99  # discount factor for future rewards
epsilon = 1.0  # initial exploration rate
epsilon_decay = 0.999  # decay rate for exploration (slower decay for NN)
epsilon_min = 0.01  # minimum exploration rate
exploration_strategy = "epsilon_greedy"

# Grid Configuration Variables
num_episodes = 3000  # number of training episodes (more episodes for NN)
grid_size_x = 40  # width of the 2D grid
grid_size_y = 40  # height of the 2D grid
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

print("Initializing Deep Q-Network Agent...")
print(f"Using device: {device}")

# Initialize environment and agent
env = GridWorld2D(
    grid_size_x=grid_size_x,
    grid_size_y=grid_size_y,
    start_pos=start_pos,
    end_pos=goal_pos,
    max_steps_per_episode=max_steps_per_episode,
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

# Initialize grid/path rich tables (always)
grid_display = plots.create_grid_display(
    grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
)
_saved_eps = agent.epsilon
agent.epsilon = 0.0
path_display = plots.display_actual_path(
    grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
)
agent.epsilon = _saved_eps

# Build initial display group including tables
grid_table = plots.grid_to_table(grid_display)
display_group = Group(
    progress,
    posProgressBar,
    stepsProgressBar,
    Panel(Text("", style="bold magenta"), title="NN Training", border_style="magenta"),
    Panel(grid_table, title="Grid (DQN)"),
    Panel(path_display, title="Current Best Path"),
)

# Initialize NN training notification state
nn_training_note = Text("", style="bold magenta")

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

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
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

            # Update grid/path rich tables periodically (not every step for performance)
            should_update_display = (
                bool(DISPLAY_STEP_INTERVAL)
                and DISPLAY_STEP_INTERVAL > 0
                and (step_count % DISPLAY_STEP_INTERVAL == 0)
            ) or done

            if should_update_display:
                # Cache Q-table once per update to avoid duplicate forward passes
                qtable_cached = agent.getQTable()
                grid_display = plots.update_grid_display(
                    grid_display, qtable_cached, start_pos, goal_pos
                )
                grid_table = plots.grid_to_table(grid_display)

                display_group = Group(
                    progress,
                    posProgressBar,
                    stepsProgressBar,
                    Panel(
                        nn_training_note, title="NN Training", border_style="magenta"
                    ),
                    Panel(grid_table, title="Grid (DQN)"),
                    # Recompute Current Best Path greedily (epsilon=0)
                    Panel(
                        plots.display_actual_path(
                            grid_size_x,
                            grid_size_y,
                            start_pos,
                            goal_pos,
                            qtable_cached,
                        ),
                        title="Current Best Path",
                    ),
                )
                live.update(display_group)

        # Store episode and step data
        episode_data.append(episode)
        step_data.append(step_count)
        epsilon_data.append(agent.epsilon)

        # Log metrics to wandb
        if USE_WANDB and plots.shouldThisEpisodeBeLogged(
            episode, num_episodes, N_IMAGE_EPISODES
        ):
            episode_duration = time.time() - episode_start_time
            best_path_length = plots.get_best_path_length(
                grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
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
            wandbconfig = {
                "episode": episode,
                "steps": step_count,
                "epsilon": agent.epsilon,
                "reward": episode_reward,
                "episode_duration": episode_duration,
                "best_path_length": best_path_length,
                "replay_buffer_size": len(agent.memory),
                "training_step": agent.training_step,
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

        # Get the Q-table for this episode (approximate from neural network)
        qtable = agent.getQTable()
        episode_qtable = {}
        for state, actions in qtable.items():
            episode_qtable[state] = {
                "up": actions["up"],
                "down": actions["down"],
                "left": actions["left"],
                "right": actions["right"],
            }

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
            _saved_eps = agent.epsilon
            agent.epsilon = 0.0
            path_display = plots.display_actual_path(
                grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
            )
            agent.epsilon = _saved_eps
            grid_display = plots.update_grid_display(
                grid_display, agent.getQTable(), start_pos, goal_pos
            )
            grid_table = plots.grid_to_table(grid_display)
            display_group = Group(
                progress,
                posProgressBar,
                stepsProgressBar,
                Panel(nn_training_note, title="NN Training", border_style="magenta"),
                Panel(grid_table, title="Grid (DQN)"),
                Panel(path_display, title="Current Best Path"),
            )
            live.update(display_group)

    end_time = time.time()

# Final display update to show completion
progress.update(task, description="DQN Training completed", completed=num_episodes)
grid_display = plots.update_grid_display(
    grid_display, agent.getQTable(), start_pos, goal_pos
)
grid_table = plots.grid_to_table(grid_display)
display_group = Group(
    progress,
    posProgressBar,
    stepsProgressBar,
    Panel(nn_training_note, title="NN Training", border_style="magenta"),
    Panel(grid_table, title="Grid (DQN)"),
    Panel(path_display, title="Current Best Path"),
)
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
    grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable()
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
print(f"Total episodes: {num_episodes}")
print(f"Final epsilon: {agent.epsilon:.4f}")
print(f"Final replay buffer size: {len(agent.memory)}")
print(f"Total training steps: {agent.training_step}")
print(f"Average steps per episode (last 100): {np.mean(step_data[-100:]):.2f}")

# Plot steps per episode : Uncomment this section to see the steps per episode
# plots.plotStepsPerEpisode(plt, episode_data, step_data)
# plt.tight_layout()
# plt.show()
