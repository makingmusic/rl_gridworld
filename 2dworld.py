import time
import matplotlib.pyplot as plt
from rich.live import Live
from rich.table import Table
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Group
from rich.panel import Panel
import plots
from q_agent2D import QLearningAgent2D
from gridworld2d import GridWorld2D


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
num_episodes = 5000  # number of training episodes
grid_size = 5  # size of the 2D grid (grid_size x grid_size)
start_pos = (0, 0)  # starting position
goal_pos = (4, 4)  # goal position


# display parameters
sleep_time = 0   # time to sleep between episodes
max_rows_in_q_value_table = 25  # Maximum number of rows to display in Q-value table


# Initialize environment and agent
env = GridWorld2D(grid_size=grid_size, start_pos=start_pos, end_pos=goal_pos)
agent = QLearningAgent2D(
    actions=["up", "down", "left", "right"],
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_min=epsilon_min,
    exploration_strategy=exploration_strategy
)

# Initialize lists to store episode and step data
episode_data = []
step_data = []
epsilon_data = []
qtable_data = []

# Initialize the display tables and progress bar
table = plots.initDisplayTable()
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
posTask = posProgressBar.add_task("Position tracking", total=grid_size * grid_size - 1)

display_group = Group(progress, posProgressBar, table)

with Live(display_group, refresh_per_second=50) as live:
    # Initialize table with one initial row
    table = plots.initDisplayTable()
    table.add_row("(0, 0)", "0.0", "0.0", "0.0", "0.0", "stay")  # Add one initial row
    live.update(table)

    # Training loop
    start_time = time.time()
    for episode in range(num_episodes):
        qtable_data.append(episode)
        state = env.reset()
        done = False
        step_count = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state
            step_count += 1

            # Update position progress bar
            current_pos = env.get_state()
            pos_value = current_pos[0] * grid_size + current_pos[1]
            posProgressBar.update(posTask, completed=pos_value)
            live.update(display_group)

        # Store episode and step data
        episode_data.append(episode)
        step_data.append(step_count)
        epsilon_data.append(agent.epsilon)

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
        table = plots.updateDisplayTableFromQTable(table, qtable, max_rows=max_rows_in_q_value_table)
        display_group = Group(progress, posProgressBar, table)
        live.update(display_group)
        time.sleep(sleep_time)

    end_time = time.time()

# Final display update to show completion
progress.update(task, description=f"Training completed", completed=num_episodes)
table = plots.updateDisplayTableFromQTable(table, agent.getQTable(), max_rows=max_rows_in_q_value_table)
display_group = Group(progress, posProgressBar, table)
live.update(display_group)
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Plot steps per episode
plots.plotStepsPerEpisode(plt, episode_data, step_data)

# Plot Q-values for selected states
states_of_interest = [(0, 0), (1,1),  (2, 2), (4, 4)]  # Example states to plot
plots.plotQTableValues(plt, qtable_data, states_of_interest)

plt.tight_layout()
plt.show() 