import time
import matplotlib.pyplot as plt
from   rich.live     import Live
from   rich.table    import Table
from   rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from   rich.console  import Group
from   rich.panel    import Panel
import plots
from   q_agent       import QLearningAgent
from   gridworld1d   import GridWorld1D

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

# default values for learning parameters
temperature         = 1.0  # default Initial temperature for softmax exploration
temperature_decay   = 0.9  # default Multiplicative factor to decay temperature each episode
temperature_min     = 0.1  # default Minimum temperature value (floor) for softmax exploration
learning_rate       = 0.1  # learning rate for Q-value updates
discount_factor     = 0.99  # discount factor for future rewards
epsilon             = 1.0  # initial exploration rate
epsilon_decay       = 0.95  # decay rate for exploration
epsilon_min         = 0.01  # minimum exploration rate

# Configuration Variables
num_episodes        = 5000 # number of training episodes
grid1DSize          = 100 # size of the 1D grid
startState          = 0 # starting state
goalState = (grid1DSize - 1)   # goal state 
optimization_strategy = "epsilon_greedy" # Options: "epsilon_greedy" or "softmax"

# display parameters
sleep_time = 0 # time to sleep between episodes
max_rows_in_q_value_table = 10  # Maximum number of rows to display in Q-value table

if (optimization_strategy == "softmax"):
    # Softmax parameters
    exploration_strategy = "softmax"  # Options: "epsilon_greedy" or "softmax"
    temperature          = 1.0  # Initial temperature for softmax exploration
    temperature_decay    = 0.9  # Multiplicative factor to decay temperature each episode
    temperature_min      = 0.1  # Minimum temperature value (floor) for softmax exploration
else:
    # Epsilon-greedy parameters
    exploration_strategy = "epsilon_greedy"  # Options: "epsilon_greedy" or "softmax"
    learning_rate        = 0.1  # learning rate for Q-value updates
    discount_factor      = 0.99  # discount factor for future rewards
    epsilon              = 1.0  # initial exploration rate
    epsilon_decay        = 0.99  # decay rate for exploration
    epsilon_min          = 0.01  # minimum exploration rate


# Init env. Init agent.
step_count = 0
env = GridWorld1D(size=grid1DSize, start_state=startState, goal_state=goalState)
agent = QLearningAgent(actions=["left", "right"], 
                      learning_rate=learning_rate, 
                      discount_factor=discount_factor,
                      epsilon=epsilon, 
                      epsilon_decay=epsilon_decay, 
                      epsilon_min=epsilon_min,
                      exploration_strategy=exploration_strategy, 
                      temperature=temperature,
                      temperature_decay=temperature_decay, 
                      temperature_min=temperature_min)

# Initialize lists to store episode and step data
episode_data     = []
step_data        = []
epsilon_data     = []
temperature_data = []  # Add list to store temperature values
qtable_data      = []  # Will store Q-tables for each episode
current_steps    = 0   # Initialize step counter variable
# Initialize the display tables and progress bar

table = plots.initDisplayTable()
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
display_group = Group(progress, stateProgressBar, table) # Create initial display group

with Live(display_group, refresh_per_second=50) as live:
    # Initialize table with initial rows
    for i in range(max_rows_in_q_value_table):
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
        temperature_data.append(agent.temperature)  # Store temperature value

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
        table = plots.updateDisplayTableFromQTable(table, qtable, max_rows=max_rows_in_q_value_table)
        display_group = Group(progress, stateProgressBar, table)
        live.update(display_group)
        time.sleep(sleep_time)
    # end for episode loop
    end_time = time.time()

#end with Live loop



# Final display update to show completion
progress.update(task, description=f"Training completed", completed=num_episodes)
live.update(display_group)
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Plot steps per episode
plots.plotStepsPerEpisode(plt, episode_data, step_data)

# Plot temperature decay
if optimization_strategy == "softmax":
    plots.plotTemperatureDecayPerEpisode(plt, episode_data, temperature_data)

# Plot Q-values for last two states
statesOfInterest = list(range(0, grid1DSize, 5)) + [grid1DSize-1]
plots.plotQTableValues(plt, qtable_data, statesOfInterest)

plt.tight_layout()
plt.show()