import io 
import time
import plots
import matplotlib.pyplot as plt
from PIL import Image
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Group, Console
from rich.panel import Panel
from q_agent2D import QLearningAgent2D
from gridworld2d import GridWorld2D
import logWandB
import pandas as pd

# wandb parameters
USE_WANDB = True  # Set to False to disable wandb logging
N_IMAGE_EPISODES = 100  # Number of intermediate episodes to log with image


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
num_episodes = 20000  # number of training episodes
grid_size_x = 50  # width of the 2D grid
grid_size_y = 50  # height of the 2D grid
start_pos = (0, 0)  # starting position at bottom left
goal_pos = (grid_size_x-1, grid_size_y-1)  # goal position at top right


# display parameters
sleep_time = 0  # time to sleep between episodes


# Initialize environment and agent
env = GridWorld2D(grid_size_x=grid_size_x, grid_size_y=grid_size_y, start_pos=start_pos, end_pos=goal_pos)
agent = QLearningAgent2D(
    actions=["up", "down", "left", "right"],
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_min=epsilon_min,
    exploration_strategy=exploration_strategy
)

# Initialize matplotlib figures and axes for all visualizations
qtable_fig, qtable_ax = plt.subplots(figsize=(grid_size_x, grid_size_y))
steps_fig, steps_ax = plt.subplots(figsize=(8, 6))
epsilon_fig, epsilon_ax = plt.subplots(figsize=(8, 6))

# Initialize lists to store episode and step data
episode_data = []
step_data = []
epsilon_data = []
qtable_data = []
last_ten_steps = [0] * 10  # Store steps from last 10 episodes

# Initialize the progress bars
progress = Progress(
    SpinnerColumn(),
    TextColumn("Training progress:"),
    BarColumn(complete_style="blue", finished_style="green"),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TaskProgressColumn(),
    transient=True
)
task = progress.add_task("Training Q-learning agent", total=num_episodes)

# Add position progress bar
posProgressBar = Progress(
    SpinnerColumn(),
    TextColumn("Current position:"),
    BarColumn(complete_style="yellow", finished_style="green"),
    MofNCompleteColumn(),
    transient=True
)
posTask = posProgressBar.add_task("Position tracking", total=grid_size_x * grid_size_y - 1)

# Add steps progress bar
stepsProgressBar = Progress(
    SpinnerColumn(),
    TextColumn("Steps in last 10 episodes:"),
    TextColumn("[bold blue]{task.description}"),
    TextColumn(" | Current:"),
    TextColumn("[bold green]{task.fields[current_steps]}"),
    transient=True
)
stepsTask = stepsProgressBar.add_task("Steps tracking", total=0, current_steps=0)

# Initialize grid display
grid_display = plots.create_grid_display(grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable())
path_display = plots.display_actual_path(grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable())

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
        "exploration_strategy": exploration_strategy
    }
    logWandB.initWandB("rl-gridworld-qlearning", config=wandbconfig)

with Live(display_group, refresh_per_second=50) as live:
    # Training loop
    start_time = time.time()
    for episode in range(num_episodes):
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
            steps_str = " | ".join(str(steps) for steps in last_ten_steps)
            stepsProgressBar.update(stepsTask, description=steps_str, current_steps=step_count)
            
            # Update grid display and show it after progress bars
            grid_display = plots.update_grid_display(grid_display, agent.getQTable(), start_pos, goal_pos)
            grid_table = plots.grid_to_table(grid_display)
                
            display_group = Group(progress, posProgressBar, stepsProgressBar,
                                  Panel(grid_table, title="Grid"),
                                  Panel(path_display, title="Current Best Path"))
            live.update(display_group)

        # Store episode and step data
        episode_data.append(episode)
        step_data.append(step_count)
        epsilon_data.append(agent.epsilon)
        # Log metrics to wandb
        if USE_WANDB and plots.shouldThisEpisodeBeLogged(episode, num_episodes, N_IMAGE_EPISODES):
            episode_duration = time.time() - episode_start_time
            best_path_length = plots.get_best_path_length(grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable())
            q_table_img, qtable_fig, qtable_ax = plots.saveQTableAsImage(
                agent.getQTableAsPolicyArrows(), 
                filename=None, 
                start_pos=start_pos, 
                goal_pos=goal_pos,
                fig=qtable_fig,
                ax=qtable_ax
            )
            q_table_df = pd.DataFrame.from_dict(
                {k: v for k, v in agent.getQTable().items()},
                orient='index'
            )
            wandbconfig = {
                "episode": episode,
                "steps": step_count,
                "epsilon": agent.epsilon,
                "reward": episode_reward,
                "q_table": logWandB.wandb.Table(dataframe=q_table_df),
                "episode_duration": episode_duration,
                "best_path_length": best_path_length,
                "q_table_heatmap_img": logWandB.wandb.Image(q_table_img)
            }
            logWandB.logEpisodeWithImageControl(wandbconfig, step=episode, episode=episode, total_episodes=num_episodes, N=N_IMAGE_EPISODES)
            
            # Update plots at the end of each episode
            steps_fig, steps_ax = plots.plotStepsPerEpisode(plt, episode_data, step_data, fig=steps_fig, ax=steps_ax)
            epsilon_fig, epsilon_ax = plots.plotEpsilonDecayPerEpisode(plt, episode_data, epsilon_data, fig=epsilon_fig, ax=epsilon_ax)
            
        # Update last ten steps
        last_ten_steps.pop(0)  # Remove oldest step count
        last_ten_steps.append(step_count)  # Add current step count

        # Get the Q-table for this episode
        qtable = agent.getQTable()
        episode_qtable = {}
        for state, actions in qtable.items():
            episode_qtable[state] = {
                'up': actions['up'],
                'down': actions['down'],
                'left': actions['left'],
                'right': actions['right']
            }
        qtable_data.append(episode_qtable)

        # Decay exploration rate at end of episode
        agent.decay_epsilon()

        # Update Displays
        progress.update(task, description=f"Episode {episode+1} of {num_episodes}", advance=1)
        live.update(display_group)
        time.sleep(sleep_time)

        # Update path display at the end of each episode
        path_display = plots.display_actual_path(grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable())
        display_group = Group(progress, posProgressBar, stepsProgressBar,
                              Panel(grid_table, title="Grid"),
                              Panel(path_display, title="Current Best Path"))
        live.update(display_group)

    end_time = time.time()

# Final display update to show completion
progress.update(task, description=f"Training completed", completed=num_episodes)
grid_display = plots.update_grid_display(grid_display, agent.getQTable(), start_pos, goal_pos)
live.update(display_group)
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Display the actual path taken by the model
path_table = plots.display_actual_path(grid_size_x, grid_size_y, start_pos, goal_pos, agent.getQTable())

if USE_WANDB:
    q_table_policyarrows = agent.getQTableAsPolicyArrows()
    q_table_img, qtable_fig, qtable_ax = plots.saveQTableAsImage(
        q_table_policyarrows, 
        "q_heatmap.png", 
        start_pos, 
        goal_pos,
        fig=qtable_fig,
        ax=qtable_ax
    )
    wandbconfig = {
        "q_table_heatmap_img": logWandB.wandb.Image(q_table_img)
    }
    logWandB.logEpisode(wandbconfig, step=num_episodes-1)

# Clean up all matplotlib figures
plt.close(qtable_fig)
plt.close(steps_fig)
plt.close(epsilon_fig)

console = Console()
console.print("\nActual path taken by the model:")
console.print(Panel(path_table, title="Path"))

if USE_WANDB:
    logWandB.closeWandB()

# Plot final visualizations
steps_fig, steps_ax = plots.plotStepsPerEpisode(plt, episode_data, step_data, fig=steps_fig, ax=steps_ax)
epsilon_fig, epsilon_ax = plots.plotEpsilonDecayPerEpisode(plt, episode_data, epsilon_data, fig=epsilon_fig, ax=epsilon_ax)

# Plot steps per episode : Uncomment this section to see the steps per episode
#plots.plotStepsPerEpisode(plt, episode_data, step_data)
#plt.tight_layout()
#plt.show() 