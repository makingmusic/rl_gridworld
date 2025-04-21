import time
import matplotlib.pyplot as plt
from rich.live     import Live
from rich.table    import Table
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console  import Group
from rich.panel    import Panel
import plots
from q_agent       import QLearningAgent
from gridworld1d   import GridWorld1D

def isTrainingComplete(qtable):
    for state, actions in qtable.items():
        if state == goalState:
            continue
        leftValue = float(list(actions.values())[0])
        rightValue = float(list(actions.values())[1])
        if rightValue <= leftValue:
            return False
    return True
#end def isTrainingComplete

def isTrainingCompleteByStepCount(stepCounterTable):
    for row in stepCounterTable.rows:
        if row.cells[1] == str(grid1DSize-1):
            return True
    return False
#end def isTrainingCompleteByStepCount



# Configuration Variables
num_episodes = 1000 # number of training episodes
grid1DSize = 100 # size of the 1D grid
startState = 0 # starting state
goalState = (grid1DSize - 1)   # goal state 
sleep_time = 0 # time to sleep between episodes

# Init env. Init agent.
step_count = 0
env = GridWorld1D(size=grid1DSize, start_state=startState, goal_state=goalState)
agent = QLearningAgent(actions=["left", "right"], learning_rate=0.1, discount_factor=0.99,
                       epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.01)

# Initialize lists to store episode and step data
episode_data = []
step_data = []
epsilon_data = []
qtable_data = []  # Will store Q-tables for each episode

# Initialize the display tables and progress bar
table = plots.initDisplayTable()
#stepCounterTable = plots.initStepCounterTable()
progress = Progress(
    SpinnerColumn(),  # Shows a spinning animation
    TextColumn("Training progress:"),  # Task description
    BarColumn(complete_style="blue", finished_style="green"),  # Progress bar
    MofNCompleteColumn(),  # Shows "M of N complete"
    TimeElapsedColumn(),  # Time elapsed
    TimeRemainingColumn(),  # Estimated time remaining
    TaskProgressColumn(),  # Task-specific progress
    transient=True
)
task = progress.add_task("Training Q-learning agent", total=num_episodes)

# Add state progress bar
stateProgressBar = Progress(
    SpinnerColumn(),
    TextColumn("Current state:"),
    BarColumn(complete_style="yellow", finished_style="green"),
    MofNCompleteColumn(),
    transient=True
)
stateTask = stateProgressBar.add_task("State tracking", total=grid1DSize-1)

# Initialize step counter variable
current_steps = 0

# Create initial display group
display_group = Group(progress, stateProgressBar, table)

with Live(display_group, refresh_per_second=50) as live:
    for i in range(grid1DSize):
        table.add_row(str(i), "0.0", "0.0", "stay")
        live.update(table)
    # end for i loop to init the display table

    # Training loop
    start_time = time.time()
    for episode in range(num_episodes):
        qtable_data.append(episode)
        state = env.reset()        # reset environment to starting state
        done = False
        step_count = 0
        
        while not done:
            action = agent.choose_action(state)             # choose action (epsilon-greedy)
            next_state, reward, done = env.step(action)     # take action, observe reward and next state
            agent.update_q_value(state, action, reward, next_state, done)  # update Q-table
            state = next_state        # move to the next state
            step_count += 1
            
            stateProgressBar.update(stateTask, completed=state) # Update state progress bar
            #updateStepCounter(stepCounterTable, episode, step_count) # todo: add this back in
            live.update(display_group)
        # end while loop
         
        # Store episode and step data
        episode_data.append(episode)
        step_data.append(step_count)
        epsilon_data.append(agent.epsilon)

        # Get the Q-table for this episode and store it in qtable_data
        qtable = agent.getQTable()
        episode_qtable = {}
        for state, actions in qtable.items():
            episode_qtable[state] = {
                'left': actions['left'],
                'right': actions['right']
            }
        qtable_data.append(episode_qtable)

        # Decay exploration rate at end of episode
        agent.decay_epsilon()
        
        # Update Displays. (progress bar and the Q-table)
        progress.update(task, description=f"Episode {episode+1} of {num_episodes}", advance=1)
        table = plots.updateDisplayTableFromQTable(table, qtable)
        #updateStepCounter(stepCounterTable, episode, step_count) # todo: add this back in
        
        # Update the display group with new plot
        display_group = Group(progress, stateProgressBar, table)
        live.update(display_group)
        time.sleep(sleep_time)
    # end for episode loop
    end_time = time.time()

#end with Live loop

# Final update to show completion
progress.update(task, description=f"Training completed", completed=num_episodes)
live.update(display_group)
print(f"Training completed in {end_time - start_time:.2f} seconds")


# plot all graphs
# plot step_count and epsilon against episode number in a line graph
#plt.figure(figsize=(plt.rcParams['figure.figsize'][0] * 0.8, plt.rcParams['figure.figsize'][1] * 0.8))

# Plot steps per episode
plots.plotStepsPerEpisode(plt, episode_data, step_data)

# Plot epsilon decay
#plots.plotEpsilonDecayPerEpisode(plt, episode_data, epsilon_data)

# Plot Q-values for last two states
statesOfInterest = list(range(0, grid1DSize, 5)) + [grid1DSize-1]
plots.plotQTableValues(plt, qtable_data, statesOfInterest)

plt.tight_layout()
plt.show()


